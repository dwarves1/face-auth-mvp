"""
main.py
───────
비대면 본인 인증 MVP – FastAPI 서버

엔드포인트:
    POST /api/v1/verify-auth   신분증 + 셀카로 본인 인증 수행

실행:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

의존성:
    pip install fastapi uvicorn python-multipart
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from functools import partial
from io import BytesIO
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from face_handler import FaceHandler, check_liveness, verify_faces

# ──────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
MAX_IMAGE_SIDE = 1280       # 긴 변을 이 픽셀 이하로 리사이즈
MAX_FILE_BYTES = 10 * 1024 * 1024   # 업로드 파일 크기 상한 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


# ══════════════════════════════════════════════
# 1. Pydantic 응답 스키마
# ══════════════════════════════════════════════
class LivenessDetail(BaseModel):
    real_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="실제 사람일 확률 [0, 1]",
        examples=[0.95],
    )
    threshold: float = Field(
        ..., description="라이브니스 판정 임계값"
    )
    passed: bool = Field(..., description="라이브니스 통과 여부")


class MatchDetail(BaseModel):
    cosine_similarity: float = Field(
        ..., ge=-1.0, le=1.0,
        description="신분증 ↔ 셀카 얼굴 코사인 유사도 [-1, 1]",
        examples=[0.82],
    )
    threshold: float = Field(
        ..., description="얼굴 매칭 판정 임계값"
    )
    passed: bool = Field(..., description="얼굴 매칭 통과 여부")


class VerifyAuthResponse(BaseModel):
    is_verified: bool = Field(
        ...,
        description="라이브니스 + 얼굴 매칭을 모두 통과하면 True",
    )
    liveness: LivenessDetail
    face_match: MatchDetail | None = Field(
        None,
        description="라이브니스 실패 시 null (Face Matching 미수행)",
    )
    elapsed_ms: float = Field(..., description="처리 소요 시간 (ms)")


class ErrorResponse(BaseModel):
    detail: str
    code: str


# ══════════════════════════════════════════════
# 2. 앱 라이프사이클 – 모델 싱글턴 초기화
# ══════════════════════════════════════════════
_face_handler: FaceHandler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 FaceHandler 를 한 번만 초기화하여 재사용."""
    global _face_handler
    logger.info("FaceHandler 초기화 시작…")
    loop = asyncio.get_event_loop()
    # 모델 로딩은 CPU 집약 작업 → 이벤트 루프 블로킹 방지를 위해 executor 사용
    _face_handler = await loop.run_in_executor(
        None,
        partial(
            FaceHandler,
            match_threshold=0.6,
            liveness_threshold=0.7,
            # antispoofing_ckpt="weights/anti_spoof_model.pth",
        ),
    )
    logger.info("FaceHandler 초기화 완료. 서버 준비.")
    yield
    logger.info("서버 종료.")


# ══════════════════════════════════════════════
# 3. FastAPI 앱 생성
# ══════════════════════════════════════════════
app = FastAPI(
    title="비대면 본인 인증 API",
    version="1.0.0",
    description="InsightFace ArcFace + Silent-Face Anti-Spoofing 기반 본인 인증 MVP",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════
# 4. 공통 유틸
# ══════════════════════════════════════════════
async def _read_and_resize(upload: UploadFile, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    """
    UploadFile → BGR NumPy 배열.

    - 파일 크기 상한 검사
    - 지원하지 않는 Content-Type 거부
    - 긴 변이 max_side 를 넘으면 비율 유지 리사이즈 (메모리 절감)
    """
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"지원하지 않는 이미지 형식입니다: {upload.content_type}. "
                   f"허용: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    raw = await upload.read()

    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"파일 크기가 너무 큽니다 ({len(raw) / 1024 / 1024:.1f} MB). "
                   f"최대 {MAX_FILE_BYTES // 1024 // 1024} MB까지 허용합니다.",
        )

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)   # BGR

    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'{upload.filename}' 이미지를 디코딩할 수 없습니다.",
        )

    # 긴 변 기준 리사이즈
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > max_side:
        scale = max_side / long_side
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(
            f"[resize] {upload.filename}: ({w}×{h}) → ({new_w}×{new_h})"
        )

    return img


def _get_handler() -> FaceHandler:
    if _face_handler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="모델이 아직 초기화되지 않았습니다. 잠시 후 다시 시도하세요.",
        )
    return _face_handler


# ══════════════════════════════════════════════
# 5. 핵심 엔드포인트
# ══════════════════════════════════════════════
@app.post(
    "/api/v1/verify-auth",
    response_model=VerifyAuthResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Liveness Detection Failed"},
        413: {"model": ErrorResponse, "description": "파일 크기 초과"},
        415: {"model": ErrorResponse, "description": "지원하지 않는 이미지 형식"},
        422: {"model": ErrorResponse, "description": "이미지 처리 실패"},
        503: {"model": ErrorResponse, "description": "모델 초기화 미완료"},
    },
    summary="비대면 본인 인증",
    description=(
        "신분증 이미지와 셀카 이미지를 업로드하면 "
        "라이브니스 탐지 → 얼굴 매칭 순으로 본인 인증을 수행합니다."
    ),
)
async def verify_auth(
    id_card: Annotated[UploadFile, File(description="신분증 이미지 (JPEG/PNG/WEBP)")],
    selfie:  Annotated[UploadFile, File(description="사용자 셀카 이미지 (JPEG/PNG/WEBP)")],
    request: Request,
) -> VerifyAuthResponse:
    """
    비즈니스 로직 Flow
    ------------------
    1. 두 이미지를 읽고 리사이즈 (동시 처리)
    2. Liveness Detection (selfie)
       - 실패 → HTTP 400 즉시 반환 (Face Matching 미수행)
    3. Face Matching (id_card ↔ selfie)
    4. is_verified = liveness.passed AND face_match.passed
    """
    start = time.perf_counter()
    handler = _get_handler()

    # ── Step 1: 이미지 로드 & 리사이즈 (두 파일 동시 await) ──
    img_id, img_selfie = await asyncio.gather(
        _read_and_resize(id_card),
        _read_and_resize(selfie),
    )

    loop = asyncio.get_event_loop()

    # ── Step 2: Liveness Detection ───────────────────────────
    # CPU/GPU 집약 연산 → run_in_executor 로 이벤트 루프 비블로킹
    try:
        liveness_result = await loop.run_in_executor(
            None,
            partial(check_liveness, img_selfie, handler),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"셀카 이미지 처리 실패: {e}",
        ) from e

    liveness_detail = LivenessDetail(
        real_probability=liveness_result.real_probability,
        threshold=liveness_result.threshold,
        passed=liveness_result.passed,
    )

    # 라이브니스 실패 → 즉시 400 반환
    if not liveness_result.passed:
        elapsed = (time.perf_counter() - start) * 1000
        logger.warning(
            f"[verify-auth] Liveness FAILED "
            f"real_prob={liveness_result.real_probability:.4f} "
            f"elapsed={elapsed:.1f}ms"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Liveness Detection Failed – "
                f"real_probability={liveness_result.real_probability:.4f} "
                f"(threshold={liveness_result.threshold})"
            ),
        )

    # ── Step 3: Face Matching ─────────────────────────────────
    try:
        match_result = await loop.run_in_executor(
            None,
            partial(verify_faces, img_id, img_selfie, handler),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"얼굴 매칭 처리 실패: {e}",
        ) from e

    match_detail = MatchDetail(
        cosine_similarity=match_result.cosine_similarity,
        threshold=match_result.threshold,
        passed=match_result.passed,
    )

    # ── Step 4: 최종 판정 ─────────────────────────────────────
    is_verified = liveness_result.passed and match_result.passed
    elapsed = (time.perf_counter() - start) * 1000

    logger.info(
        f"[verify-auth] is_verified={is_verified} "
        f"liveness={liveness_result.real_probability:.4f} "
        f"similarity={match_result.cosine_similarity:.4f} "
        f"elapsed={elapsed:.1f}ms"
    )

    return VerifyAuthResponse(
        is_verified=is_verified,
        liveness=liveness_detail,
        face_match=match_detail,
        elapsed_ms=round(elapsed, 2),
    )


# ══════════════════════════════════════════════
# 6. 전역 예외 핸들러
# ══════════════════════════════════════════════
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"처리되지 않은 예외 발생: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "서버 내부 오류가 발생했습니다.", "code": "INTERNAL_SERVER_ERROR"},
    )


# ══════════════════════════════════════════════
# 7. 헬스체크
# ══════════════════════════════════════════════
@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {
        "status": "ok",
        "model_ready": _face_handler is not None,
    }
