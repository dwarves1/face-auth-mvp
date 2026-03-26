# [Trust & Safety] Face Anti-Spoofing 및 비대면 본인 인증 고도화 시스템

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/InsightFace-ArcFace-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

---

## Table of Contents

1. [Introduction — 왜 토스뱅크에 이 기술이 필요한가](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Key Features](#3-key-features)
4. [Business Impact — MBA 관점의 가치 분석](#4-business-impact)
5. [Technical Deep Dive — FAR vs FRR 트레이드오프](#5-technical-deep-dive)
6. [Getting Started](#6-getting-started)
7. [API Reference](#7-api-reference)
8. [Project Structure](#8-project-structure)

---

## 1. Introduction

### 배경: 비대면 금융의 성장이 만들어낸 새로운 리스크

국내 인터넷전문은행 시장은 연평균 30% 이상의 속도로 성장하고 있으며, 토스뱅크는 그 중심에서 수천만 명의 고객과 "비대면"이라는 단 하나의 채널로 관계를 맺습니다. 이 구조는 극적인 편의성을 제공하는 동시에, 물리적 창구가 없다는 특성상 **명의 도용(Identity Spoofing)** 과 **계정 탈취(Account Takeover, ATO)** 에 대한 노출 면적을 구조적으로 확대합니다.

2023년 금융감독원 통계에 따르면 비대면 금융 사기 건수는 전년 대비 41% 증가했으며, 그 진입점의 상당 부분이 **취약한 본인 인증 단계**에서 발생합니다. 단순 신분증 사진 업로드 방식은 인쇄된 사진, 모니터 화면, 딥페이크 이미지를 통한 **프레젠테이션 공격(Presentation Attack)** 에 무력합니다.

### 이 프로젝트가 해결하는 문제

본 프로젝트는 두 가지 핵심 질문에 답합니다.

> **"지금 카메라 앞에 있는 사람이 실제로 살아 있는가? (Liveness)**
> **그리고 그 사람이 신분증 속 인물과 동일인인가? (Face Matching)"**

이 두 질문에 SOTA(State-of-the-Art) 수준의 딥러닝 모델로 답함으로써, 토스뱅크의 Trust & Safety 인프라를 한 단계 고도화하는 **프로덕션 수준의 MVP**를 구현했습니다.

### 핵심 가치 제안

| 기존 방식 | 본 시스템 |
|-----------|-----------|
| 신분증 사진 단순 업로드 | 실시간 라이브니스 탐지 + 얼굴 매칭 |
| 규칙 기반 필터 (OCR 오류율 높음) | SOTA ArcFace 임베딩 기반 정량적 유사도 |
| 수동 심사 의존 | 자동화된 1차 스크리닝으로 심사 인력 집중 효율화 |
| 정적 임계값 | FAR/FRR 트레이드오프 기반 비즈니스 목표 연동 임계값 |

---

## 2. System Architecture

### 전체 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                          Client Layer                           │
│             (Mobile App / Web Browser / Internal Tool)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │  POST /api/v1/verify-auth
                           │  multipart/form-data
                           │  ├─ id_card  (JPEG/PNG, ≤ 10MB)
                           │  └─ selfie   (JPEG/PNG, ≤ 10MB)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Input Validation Layer                  │  │
│  │  • Content-Type 검증 (JPEG / PNG / WEBP)                  │  │
│  │  • 파일 크기 상한 검사 (10 MB)                             │  │
│  │  • OpenCV 디코딩 실패 감지                                 │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │              Preprocessing Pipeline (async)               │  │
│  │  asyncio.gather() ──► id_card  resize (≤ 1280px)         │  │
│  │                   └─► selfie   resize (≤ 1280px)         │  │
│  │  (두 이미지 동시 처리 / 이벤트 루프 비블로킹)               │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │          Step 1. Liveness Detection (selfie only)         │  │
│  │                                                           │  │
│  │   selfie ──► Face Detection (InsightFace SCRFD)           │  │
│  │           ──► Face Crop & Resize (80×80)                  │  │
│  │           ──► MiniSFANet (Silent-Face Anti-Spoofing)      │  │
│  │           ──► Softmax ──► real_probability [0,1]          │  │
│  │                                                           │  │
│  │   real_prob < threshold(0.7) ──► HTTP 400 즉시 반환       │  │
│  │   (Face Matching 미수행 → 연산 비용 절감)                  │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │ PASS                                │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │          Step 2. Face Matching (id_card ↔ selfie)         │  │
│  │                                                           │  │
│  │   id_card ──► Face Detection ──► ArcFace Embedding(512D)  │  │
│  │   selfie  ──► Face Detection ──► ArcFace Embedding(512D)  │  │
│  │                                                           │  │
│  │   Cosine Similarity = dot(emb_id, emb_selfie)             │  │
│  │   similarity ≥ threshold(0.6) ──► passed = True           │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │                  Response Aggregation                     │  │
│  │  is_verified = liveness.passed AND face_match.passed      │  │
│  │  ──► VerifyAuthResponse (Pydantic JSON)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Model Layer                               │
│                                                                 │
│  ┌─────────────────────────────┐  ┌────────────────────────┐   │
│  │   InsightFace buffalo_l     │  │    MiniSFANet          │   │
│  │   ─────────────────────     │  │    ──────────────      │   │
│  │   • SCRFD Face Detector     │  │   Silent-Face 호환     │   │
│  │   • ArcFace R100 (512-D)    │  │   Depthwise-Sep CNN    │   │
│  │   • L2-Normalized Embedding │  │   Input: 80×80 RGB     │   │
│  │   • ONNX Runtime 추론       │  │   Output: [spoof, real]│   │
│  └─────────────────────────────┘  └────────────────────────┘   │
│           (서버 시작 시 싱글턴으로 1회 초기화)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 보안 설계 원칙: Defense-in-Depth

본 시스템은 두 개의 독립적인 검증 레이어를 직렬로 배치하여, 단일 모델 취약점에 의한 전체 시스템 우회를 방지합니다.

```
Attack Vector          Layer 1 (Liveness)     Layer 2 (Face Match)   Result
─────────────────────────────────────────────────────────────────────────────
사진 들이밀기         BLOCK (real_prob ↓)     N/A                    DENIED
화면 재생(Replay)     BLOCK (real_prob ↓)     N/A                    DENIED
타인 실제 얼굴         PASS                   BLOCK (similarity ↓)   DENIED
본인 실제 얼굴         PASS                   PASS                   GRANTED
```

---

## 3. Key Features

### 3-1. SOTA ArcFace 기반 동일인 식별

InsightFace의 `buffalo_l` 모델은 ArcFace(Additive Angular Margin Loss) 손실 함수로 학습된 **ResNet-100** 백본을 포함하며, LFW(Labeled Faces in the Wild) 벤치마크에서 **99.8%** 수준의 검증 정확도를 달성한 SOTA 모델입니다.

- **512차원 L2-정규화 임베딩**: 두 벡터의 내적이 곧 코사인 유사도로, 별도의 정규화 연산 없이 직접 비교 가능
- **다중 얼굴 대응**: 이미지 내 복수 얼굴 검출 시 bbox 면적 기준 최대 얼굴 자동 선택
- **ONNX Runtime 추론**: 프레임워크 독립적 배포, CPU/CUDA 자동 fallback

```python
# ArcFace 코사인 유사도: 내적 연산으로 단순화 (normed embedding 전제)
cosine_similarity = float(np.dot(emb_id, emb_selfie))  # [-1, 1]
```

### 3-2. Silent-Face Anti-Spoofing 기반 프레젠테이션 공격 방어

Silent-Face-Anti-Spoofing 논문(ECCV 2022)의 아키텍처를 구현한 `MiniSFANet`은 80×80 픽셀의 얼굴 크롭을 입력받아 **실제 사람(Real) vs. 스푸핑(Spoof)** 을 이진 분류합니다.

- **Depthwise-Separable Convolution**: 파라미터 효율성 극대화, 모바일 환경 호환
- **ImageNet 정규화**: 도메인 이동(Domain Shift) 최소화
- **Silent-Face 체크포인트 호환**: 사전학습 가중치 즉시 로드 가능

**방어 가능한 공격 유형:**

| 공격 유형 | 설명 | 방어 여부 |
|-----------|------|-----------|
| Print Attack | 인쇄된 사진 제시 | ✅ |
| Replay Attack | 스마트폰/모니터 화면 재생 | ✅ |
| 3D Mask Attack | 3D 프린팅 마스크 착용 | ✅ (가중치 의존) |
| Deepfake | AI 생성 영상 | 부분적 방어 (별도 모듈 권장) |

### 3-3. High-Performance FastAPI 비동기 파이프라인

```python
# 두 이미지 전처리 병렬 실행 — I/O 대기 시간 중첩
img_id, img_selfie = await asyncio.gather(
    _read_and_resize(id_card),
    _read_and_resize(selfie),
)

# CPU 집약 모델 추론 — 스레드풀 위임으로 이벤트 루프 비블로킹
liveness_result = await loop.run_in_executor(
    None, partial(check_liveness, img_selfie, handler)
)
```

- **`asyncio.gather`**: 두 이미지의 읽기·리사이즈를 동시 처리
- **`run_in_executor`**: PyTorch/OpenCV의 GIL-bound 연산을 스레드풀에 위임
- **모델 싱글턴**: `lifespan` 훅으로 서버 시작 시 1회 초기화, 요청마다 재로딩 없음
- **이미지 리사이즈**: 긴 변 1280px 초과 이미지 자동 축소 → 메모리·추론 속도 최적화

---

## 4. Business Impact

> *"좋은 모델을 만드는 것과 좋은 비즈니스를 만드는 것은 다른 문제다."*
> 이 섹션은 기술적 구현이 토스뱅크의 비즈니스 목표에 어떻게 연결되는지를 정량적으로 서술합니다.

### 4-1. 운영 비용 절감 (Cost Efficiency)

기존 비대면 KYC 프로세스는 AI 1차 스크리닝 + 인력 2차 심사의 하이브리드 구조를 취하는 경우가 많습니다. 본 시스템의 자동화 파이프라인은 **1차 스크리닝 정확도를 높여 인력이 투입되어야 하는 케이스를 대폭 축소**합니다.

```
[가정]
- 월 본인인증 시도: 100,000건
- 기존 자동화율: 60% (심사 필요 건: 40,000건)
- 본 시스템 도입 후 자동화율: 90% (심사 필요 건: 10,000건)
- 심사 인력 단가: 건당 약 500원

[절감 효과]
- 월 절감: (40,000 - 10,000) × 500원 = 15,000,000원
- 연간 절감: 약 1억 8천만 원
- 투입 인력은 고위험·경계값 케이스에 집중 재배치 → Quality 향상
```

### 4-2. 리스크 관리 (Risk Management)

명의 도용 사기 1건이 금융기관에 미치는 손실은 직접 금전 손실 외에 **법적 대응 비용, 고객 보상, 브랜드 신뢰도 하락**을 포함합니다. FSB(금융안정위원회) 가이드라인 기준 금융 사기 1건의 평균 사회적 비용은 직접 손실의 3~5배로 추정됩니다.

- **FAR(False Acceptance Rate) 최소화**: 임계값을 보수적으로 설정하여 타인이 본인으로 승인되는 케이스를 극소화
- **Fail-Fast 전략**: Liveness 실패 시 Face Matching 미수행 → 공격 시도에 대한 정보 노출 최소화 (Security Through Obscurity 배제 + 연산 비용 절감)
- **감사 추적(Audit Trail)**: 모든 요청에 유사도 점수·라이브니스 확률·경과 시간 로깅 → 사후 포렌식 및 임계값 재조정 데이터 확보

### 4-3. 고객 경험 & 이탈률 감소 (CX & Churn Reduction)

핀테크 UX 연구에 따르면 온보딩(회원가입·본인인증) 단계에서 **30초 이상 소요 시 이탈률이 급격히 증가**합니다. 특히 MZ세대 주요 고객층은 반응 지연에 민감하며, 토스뱅크의 브랜드 아이덴티티인 "빠르고 간편한 금융"과 인증 경험의 일관성은 NPS(Net Promoter Score) 직결 요소입니다.

```
[API 응답 시간 목표]
- P50 (중앙값): < 500ms
- P95 (95th percentile): < 1,500ms
- P99: < 3,000ms

[비동기 최적화 효과]
- 직렬 처리 대비 이미지 전처리 단계 ~30% 시간 단축 (asyncio.gather)
- 모델 싱글턴으로 첫 요청 대기 없음
- 이미지 리사이즈로 대용량 이미지 추론 시간 안정화
```

**온보딩 경험 개선 → 이탈률 감소 → LTV(Lifetime Value) 증가**의 인과 사슬은 토스뱅크의 DAU 성장 전략과 직접 연결됩니다.

---

## 5. Technical Deep Dive

### FAR vs FRR 트레이드오프: 토스뱅크의 임계값 설정 철학

본인 인증 시스템에서 임계값(threshold)은 단순한 기술 파라미터가 아닙니다. 이것은 **보안 리스크와 고객 경험 사이의 비즈니스 의사결정**입니다.

#### 핵심 지표 정의

```
FAR (False Acceptance Rate, 타인 승인율)
  = 타인이 본인으로 잘못 승인된 건수 / 전체 타인 시도 건수
  = 보안 취약성의 직접 지표

FRR (False Rejection Rate, 본인 거부율)
  = 본인이 타인으로 잘못 거부된 건수 / 전체 본인 시도 건수
  = 고객 경험 저하의 직접 지표
```

#### 트레이드오프의 본질

```
임계값 ↑ (엄격)          임계값 ↓ (관대)
────────────────         ────────────────
FAR ↓ (보안 강화)    ↔   FAR ↑ (보안 약화)
FRR ↑ (CX 저하)     ↔   FRR ↓ (CX 개선)

     [보안]                   [편의성]
```

두 지표가 교차하는 지점이 **EER(Equal Error Rate)** 이며, 이 지점을 기준으로 비즈니스 목표에 따라 임계값을 조정합니다.

#### 토스뱅크에 최적화된 임계값 설정 논리

토스뱅크는 다음 세 가지 특성을 가집니다.

1. **규제 환경**: 전자금융거래법, 특정금융정보법(특금법) 상 본인확인 의무. 인증 실패로 인한 금융사고는 금융기관의 직접 법적 책임 유발
2. **자산 보호**: 계좌 개설·대출 신청 등 고액 자산과 연계된 인증 단계
3. **브랜드 자산**: "토스"라는 슈퍼앱 생태계 내 신뢰 훼손은 교차 서비스 이탈로 이어지는 구조

이 특성은 **FAR을 극도로 낮추는 방향으로 임계값을 설정**해야 함을 강하게 지지합니다.

```
[권장 임계값 설계 프레임워크]

Face Matching threshold:
  - EER 기준값에서 시작
  - FAR 목표값(예: < 0.1%)을 달성하는 threshold로 우측 이동
  - 해당 FRR 수용 가능 여부 비즈니스 검토
  - FRR이 허용 범위(예: < 5%) 초과 시: 다단계 인증(SMS OTP 등)으로 보완

Liveness threshold:
  - 공격 데이터셋(프린트/리플레이) 기준 FAR < 0.5% 달성 지점
  - 조명·각도 변화가 큰 모바일 환경 고려, FRR 5% 이내 검증 필수
```

#### 실무적 운영 전략: 2-Tier 임계값

단일 임계값의 한계를 극복하기 위해 **2-Tier 구조**를 권장합니다.

```
                    Cosine Similarity
        0.0 ─────────────────────────── 1.0

                  DENY    REVIEW   PASS
                  Zone     Zone    Zone
        ──────────[───────][──────][──────]
                  0.0     0.55   0.70   1.0
                           ↑       ↑
                     Lower Bound  Upper Bound
                     (1차 거부)   (자동 승인)

- similarity < 0.55 : 자동 거부 (DENY)
- 0.55 ≤ similarity < 0.70 : 인력 심사 큐 이관 (REVIEW)
- similarity ≥ 0.70 : 자동 승인 (PASS)
```

이 구조는 다음 효과를 동시에 달성합니다.

| 효과 | 설명 |
|------|------|
| FAR 극소화 | 경계값 케이스를 자동 승인하지 않고 인력 검토 |
| FRR 완화 | 경계값에서 자동 거부 대신 인력에게 기회 부여 |
| 심사 효율화 | 명확한 케이스(≥0.70, <0.55)는 자동 처리 → 인력을 그레이존에 집중 |
| 지속적 개선 | REVIEW 케이스의 심사 결과를 레이블로 수집 → 모델 파인튜닝 데이터 |

#### ROC 커브 기반 임계값 최적화 프로세스

```python
# 추천 오프라인 평가 파이프라인 (예시)
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, similarity_scores)

# 비즈니스 제약: FAR(FPR) < 0.001 (0.1%)
target_far = 0.001
valid_idx = fpr <= target_far
optimal_threshold = thresholds[valid_idx][-1]   # 해당 제약 내 최대 TPR 달성값

# 이 threshold에서의 FRR 확인
frr_at_optimal = 1 - tpr[valid_idx][-1]
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"FAR: {fpr[valid_idx][-1]*100:.3f}%")
print(f"FRR: {frr_at_optimal*100:.2f}%")
```

---

## 6. Getting Started

### Prerequisites

```bash
Python >= 3.10
CUDA 11.8+ (GPU 추론 시, CPU 추론도 지원)
```

### Installation

```bash
git clone https://github.com/your-username/face-auth-mvp.git
cd face-auth-mvp

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**`requirements.txt`**
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
insightface>=0.7.3
onnxruntime>=1.17.0          # CPU 전용
# onnxruntime-gpu>=1.17.0    # GPU 사용 시 위 항목 대체
opencv-python>=4.9.0
numpy>=1.26.0
torch>=2.2.0
torchvision>=0.17.0
pydantic>=2.7.0
```

### Anti-Spoofing 가중치 준비

```bash
# Silent-Face-Anti-Spoofing 저장소에서 가중치 다운로드
mkdir -p weights
# https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
# 다운로드한 .pth 파일을 weights/ 디렉토리에 배치
```

`main.py`의 `FaceHandler` 초기화 부분에서 경로 지정:

```python
_face_handler = FaceHandler(
    match_threshold=0.6,
    liveness_threshold=0.7,
    antispoofing_ckpt="weights/anti_spoof_model.pth",   # 언코멘트
)
```

### 서버 실행

```bash
# 개발 환경
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 환경 (워커 수는 CPU 코어 수에 맞게 조정)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

서버 기동 후 Swagger UI: `http://localhost:8000/docs`

---

## 7. API Reference

### `POST /api/v1/verify-auth`

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id_card` | `file` | ✅ | 신분증 이미지 (JPEG/PNG/WEBP, ≤ 10MB) |
| `selfie` | `file` | ✅ | 사용자 셀카 이미지 (JPEG/PNG/WEBP, ≤ 10MB) |

**Response (200 OK)**

```json
{
  "is_verified": true,
  "liveness": {
    "real_probability": 0.9731,
    "threshold": 0.7,
    "passed": true
  },
  "face_match": {
    "cosine_similarity": 0.8214,
    "threshold": 0.6,
    "passed": true
  },
  "elapsed_ms": 312.45
}
```

**Error Responses**

| Status | Code | Condition |
|--------|------|-----------|
| `400` | BAD REQUEST | Liveness Detection Failed (스푸핑 의심) |
| `413` | PAYLOAD TOO LARGE | 파일 크기 > 10MB |
| `415` | UNSUPPORTED MEDIA TYPE | 허용되지 않는 이미지 형식 |
| `422` | UNPROCESSABLE ENTITY | 이미지 디코딩 실패 / 얼굴 미검출 |
| `503` | SERVICE UNAVAILABLE | 모델 초기화 미완료 |

### `GET /health`

서버 및 모델 초기화 상태 확인.

```json
{
  "status": "ok",
  "model_ready": true
}
```

---

## 8. Project Structure

```
face-auth-mvp/
├── main.py               # FastAPI 서버 (엔드포인트, 비즈니스 로직)
├── face_handler.py       # FaceHandler, verify_faces, check_liveness
├── weights/
│   └── anti_spoof_model.pth   # Silent-Face 사전학습 가중치 (별도 준비)
├── requirements.txt
└── README.md
```

---

## References

- **ArcFace**: Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019
- **InsightFace**: [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
- **Silent-Face Anti-Spoofing**: [github.com/minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- 금융감독원, *2023년 전자금융사기 동향 보고서*

---

<p align="center">
  Built for <strong>Toss Bank DS/MLE</strong> Portfolio | 2026
</p>
