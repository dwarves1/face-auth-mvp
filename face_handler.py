"""
face_handler.py
───────────────
비대면 본인 인증 MVP – 얼굴 매칭 & 라이브니스 탐지 모듈

의존성:
    pip install insightface onnxruntime opencv-python numpy torch torchvision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 타입 alias
# ──────────────────────────────────────────────
ImageInput = Union[str, Path, np.ndarray]   # 파일 경로 또는 BGR NumPy 배열


# ──────────────────────────────────────────────
# 결과 데이터클래스
# ──────────────────────────────────────────────
@dataclass
class MatchResult:
    cosine_similarity: float    # [-1, 1] – 높을수록 동일인
    passed: bool                # 임계값 이상이면 True
    threshold: float


@dataclass
class LivenessResult:
    real_probability: float     # [0, 1] – 높을수록 실제 사람
    passed: bool                # 임계값 이상이면 True
    threshold: float


# ══════════════════════════════════════════════
# 1. FaceHandler – 모델 초기화 & 공통 유틸
# ══════════════════════════════════════════════
class FaceHandler:
    """
    InsightFace ArcFace + Silent-Face Anti-Spoofing 래퍼.

    Parameters
    ----------
    match_threshold   : 얼굴 매칭 코사인 유사도 임계값 (default 0.6)
    liveness_threshold: 라이브니스 실제 사람 확률 임계값 (default 0.7)
    det_size          : InsightFace 얼굴 검출 해상도
    device            : 'cuda' | 'cpu' (None 이면 자동 선택)
    antispoofing_ckpt : Silent-Face 체크포인트 경로 (None 이면 경량 CNN 사용)
    """

    def __init__(
        self,
        match_threshold: float = 0.6,
        liveness_threshold: float = 0.7,
        det_size: tuple[int, int] = (640, 640),
        device: str | None = None,
        antispoofing_ckpt: str | Path | None = None,
    ) -> None:
        self.match_threshold = match_threshold
        self.liveness_threshold = liveness_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── InsightFace 초기화 ──────────────────
        logger.info("InsightFace(ArcFace) 모델 로딩 중…")
        self._face_app = FaceAnalysis(
            name="buffalo_l",           # ArcFace R100 기반 512-D 임베딩
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            ),
        )
        self._face_app.prepare(ctx_id=0 if self.device == "cuda" else -1,
                               det_size=det_size)
        logger.info("InsightFace 로딩 완료.")

        # ── Anti-Spoofing 모델 초기화 ───────────
        logger.info("Anti-Spoofing 모델 로딩 중…")
        self._liveness_model = self._build_liveness_model(antispoofing_ckpt)
        self._liveness_model.eval().to(self.device)
        logger.info("Anti-Spoofing 모델 로딩 완료.")

    # ──────────────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────────────
    @staticmethod
    def _load_image(source: ImageInput) -> np.ndarray:
        """파일 경로 또는 NumPy BGR 배열을 BGR uint8 배열로 반환."""
        if isinstance(source, np.ndarray):
            img = source.copy()
        else:
            img = cv2.imread(str(source))
            if img is None:
                raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {source}")
        if img.ndim == 2:                       # 그레이스케일 → BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        이미지에서 얼굴을 검출하고 ArcFace 임베딩(512-D)을 반환.
        얼굴이 없으면 ValueError 를 발생시킨다.
        """
        faces = self._face_app.get(image)
        if not faces:
            raise ValueError("이미지에서 얼굴을 검출하지 못했습니다.")
        # 검출된 얼굴 중 bbox 면적이 가장 큰 얼굴 선택
        main_face = max(faces, key=lambda f: (
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ))
        return main_face.normed_embedding          # L2-정규화된 512-D 벡터

    # ──────────────────────────────────────────
    # Anti-Spoofing 모델 빌드
    # ──────────────────────────────────────────
    def _build_liveness_model(
        self, ckpt_path: str | Path | None
    ) -> nn.Module:
        """
        Silent-Face-Anti-Spoofing 체크포인트가 제공되면 로드하고,
        없으면 동일한 아키텍처의 경량 MiniSFANet 을 사용.

        실제 배포 시에는 아래 저장소의 가중치를 사용하세요:
        https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
        """
        model = MiniSFANet()
        if ckpt_path is not None:
            state = torch.load(str(ckpt_path), map_location="cpu")
            # Silent-Face 원본 저장소 키 구조에 맞게 로드
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Anti-Spoofing 체크포인트 로드: {ckpt_path}")
        else:
            logger.warning(
                "antispoofing_ckpt 미지정 – 무작위 가중치의 MiniSFANet 사용. "
                "실제 서비스에서는 사전학습 가중치를 반드시 지정하세요."
            )
        return model

    @staticmethod
    def _preprocess_for_liveness(face_crop: np.ndarray) -> torch.Tensor:
        """
        얼굴 크롭 BGR → (1, 3, 80, 80) float32 텐서.
        Silent-Face 논문 기준 입력 크기.
        """
        img = cv2.resize(face_crop, (80, 80))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    def _crop_face(self, image: np.ndarray) -> np.ndarray:
        """가장 큰 얼굴의 bbox 를 이미지에서 잘라 반환."""
        faces = self._face_app.get(image)
        if not faces:
            raise ValueError("이미지에서 얼굴을 검출하지 못했습니다.")
        face = max(faces, key=lambda f: (
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ))
        x1, y1, x2, y2 = map(int, face.bbox)
        # 경계 클리핑
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]


# ══════════════════════════════════════════════
# 2. verify_faces – 얼굴 매칭
# ══════════════════════════════════════════════
def verify_faces(
    id_image: ImageInput,
    selfie_image: ImageInput,
    handler: FaceHandler,
) -> MatchResult:
    """
    신분증 이미지와 셀카 이미지의 얼굴이 동일인인지 판별합니다.

    Parameters
    ----------
    id_image     : 신분증 이미지 (파일 경로 또는 BGR NumPy 배열)
    selfie_image : 사용자 셀카 이미지
    handler      : 초기화된 FaceHandler 인스턴스

    Returns
    -------
    MatchResult
        - cosine_similarity : 두 임베딩의 코사인 유사도 [-1, 1]
        - passed            : 유사도 >= match_threshold 이면 True
        - threshold         : 적용된 임계값
    """
    img_id     = FaceHandler._load_image(id_image)
    img_selfie = FaceHandler._load_image(selfie_image)

    emb_id     = handler._get_embedding(img_id)       # (512,)
    emb_selfie = handler._get_embedding(img_selfie)   # (512,)

    # 코사인 유사도 – normed_embedding 은 이미 L2 정규화돼 있으므로 내적 = 코사인
    cosine_sim: float = float(np.dot(emb_id, emb_selfie))

    result = MatchResult(
        cosine_similarity=cosine_sim,
        passed=cosine_sim >= handler.match_threshold,
        threshold=handler.match_threshold,
    )
    logger.info(
        f"[FaceMatch] cosine={cosine_sim:.4f} "
        f"threshold={handler.match_threshold} passed={result.passed}"
    )
    return result


# ══════════════════════════════════════════════
# 3. check_liveness – 라이브니스 탐지
# ══════════════════════════════════════════════
def check_liveness(
    selfie_image: ImageInput,
    handler: FaceHandler,
) -> LivenessResult:
    """
    셀카 이미지가 실제 사람인지, 스푸핑 시도인지 판별합니다.

    Parameters
    ----------
    selfie_image : 사용자 셀카 이미지 (파일 경로 또는 BGR NumPy 배열)
    handler      : 초기화된 FaceHandler 인스턴스

    Returns
    -------
    LivenessResult
        - real_probability : 실제 사람일 확률 [0, 1]
        - passed           : 확률 >= liveness_threshold 이면 True
        - threshold        : 적용된 임계값
    """
    img = FaceHandler._load_image(selfie_image)
    face_crop = handler._crop_face(img)

    tensor = FaceHandler._preprocess_for_liveness(face_crop).to(handler.device)

    with torch.no_grad():
        logits = handler._liveness_model(tensor)        # (1, 2)
        probs  = F.softmax(logits, dim=1)               # [spoof_prob, real_prob]
        real_prob: float = probs[0, 1].item()

    result = LivenessResult(
        real_probability=real_prob,
        passed=real_prob >= handler.liveness_threshold,
        threshold=handler.liveness_threshold,
    )
    logger.info(
        f"[Liveness] real_prob={real_prob:.4f} "
        f"threshold={handler.liveness_threshold} passed={result.passed}"
    )
    return result


# ══════════════════════════════════════════════
# 4. MiniSFANet – Silent-Face 호환 경량 CNN
# ══════════════════════════════════════════════
class _DepthwiseSeparable(nn.Module):
    """Depthwise-Separable Convolution Block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class MiniSFANet(nn.Module):
    """
    Silent-Face-Anti-Spoofing 논문과 호환되는 경량 MobileNet 계열 분류기.

    입력  : (B, 3, 80, 80)  – ImageNet 정규화된 RGB 텐서
    출력  : (B, 2)           – [spoof_logit, real_logit]

    실제 학습된 가중치 없이는 무작위 예측만 가능합니다.
    학습된 가중치는 아래를 참고하세요:
    https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            _DepthwiseSeparable(32,  64, stride=1),
            _DepthwiseSeparable(64,  128, stride=2),
            _DepthwiseSeparable(128, 128, stride=1),
            _DepthwiseSeparable(128, 256, stride=2),
            _DepthwiseSeparable(256, 256, stride=1),
            _DepthwiseSeparable(256, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ══════════════════════════════════════════════
# 5. 간단한 사용 예시 (직접 실행 시)
# ══════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    handler = FaceHandler(
        match_threshold=0.6,
        liveness_threshold=0.7,
        # antispoofing_ckpt="weights/anti_spoof_model.pth",  # 실제 가중치 경로
    )

    # ── 얼굴 매칭 테스트 ─────────────────────
    # match = verify_faces("id_card.jpg", "selfie.jpg", handler)
    # print(f"[FaceMatch] similarity={match.cosine_similarity:.4f}, passed={match.passed}")

    # ── 라이브니스 테스트 ────────────────────
    # liveness = check_liveness("selfie.jpg", handler)
    # print(f"[Liveness ] real_prob={liveness.real_probability:.4f}, passed={liveness.passed}")
