# =============================================================================
# 사업자등록증 OCR 웹 애플리케이션 - 경량버전 (OpenAI Vision API)
# =============================================================================
# EasyOCR 대신 OpenAI GPT-4o Vision API를 사용해 OCR + 항목 추출을 수행한다.
#
# 장점:
#   - EasyOCR / OpenCV / numpy 등 무거운 패키지 불필요
#   - 모델 다운로드 없이 즉시 실행 가능
#   - GPT-4o가 OCR과 항목 추출을 한 번에 처리 → 정규식 불필요
#   - 기울어진 사진, 저화질 이미지에도 강인함
#
# 단점:
#   - OpenAI API Key 필요 (유료, 호출당 과금)
#   - 인터넷 연결 필수
#   - 개인정보가 외부 서버로 전송됨 → 실제 사업자등록증 사용 시 주의
#
# 필요 패키지 (기존 대비 대폭 축소):
#   pip install streamlit openai pillow pandas openpyxl
#
# 실행:
#   streamlit run business_license_ocr_app_light.py
# =============================================================================

# ── 표준 라이브러리 ──────────────────────────────────────────────────────────
import base64           # 이미지를 Base64로 인코딩해 API에 전달
import json             # GPT 응답 JSON 파싱 및 결과 저장
import tempfile         # 업로드 파일을 임시 경로에 저장
from datetime import datetime   # 저장 파일명 타임스탬프
from io import BytesIO          # 엑셀 파일 메모리 생성
from pathlib import Path        # OS 독립 파일 경로 처리

# ── 서드파티 라이브러리 ───────────────────────────────────────────────────────
import fitz                     # PyMuPDF - PDF를 이미지로 변환
import pandas as pd             # 표 데이터 처리 및 Excel 저장
import streamlit as st          # 웹 UI 프레임워크
from openai import OpenAI       # OpenAI Python SDK v1.x
from PIL import Image           # 이미지 열기 / 미리보기


# =============================================================================
# ── 상수 / 경로 설정
# =============================================================================

APP_DIR    = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# UI 표에 표시할 항목 순서
FIELD_NAMES = [
    "사업자등록번호",
    "상호",
    "대표자",
    "개업연월일",
    "사업장소재지",
    "업태",
    "종목",
]

# GPT-4o에게 전달할 추출 지시 프롬프트
# JSON만 반환하도록 엄격히 지시해 파싱 오류를 방지한다.
EXTRACT_PROMPT = """
아래는 한국 사업자등록증 이미지입니다.
이미지에서 다음 7개 항목을 정확히 추출해 JSON 형식으로만 응답하세요.
마크다운 코드블록(```)이나 설명 문구 없이 순수 JSON만 출력하세요.

추출 항목:
- 사업자등록번호 (형식: XXX-XX-XXXXX)
- 상호
- 대표자
- 개업연월일
- 사업장소재지
- 업태
- 종목

응답 형식 예시:
{
  "사업자등록번호": "123-45-67890",
  "상호": "홍길동상사",
  "대표자": "홍길동",
  "개업연월일": "2010.03.15",
  "사업장소재지": "서울특별시 강남구 테헤란로 123",
  "업태": "도소매",
  "종목": "전자제품"
}

인식이 불가능한 항목은 빈 문자열("")로 채우세요.
""".strip()


# =============================================================================
# ── Streamlit 페이지 설정
# =============================================================================

st.set_page_config(
    page_title="사업자등록증 OCR (경량)",
    page_icon="🏢",
    layout="wide",
)


# =============================================================================
# ── OpenAI 클라이언트 초기화
# =============================================================================

def get_client(api_key: str) -> OpenAI:
    """
    입력받은 API Key로 OpenAI 클라이언트를 생성해 반환한다.

    Args:
        api_key (str): 사용자가 입력한 OpenAI API Key

    Returns:
        OpenAI: 초기화된 클라이언트 객체
    """
    return OpenAI(api_key=api_key)


# =============================================================================
# ── 이미지 → Base64 변환
# =============================================================================

def pdf_to_image(pdf_path: Path) -> Path:
    """
    PDF 첫 페이지를 PNG 이미지로 변환해 임시 파일로 저장한다.
    사업자등록증은 단일 페이지이므로 첫 페이지만 처리한다.

    Args:
        pdf_path (Path): PDF 파일 경로

    Returns:
        Path: 변환된 PNG 임시 파일 경로
    """
    doc = fitz.open(str(pdf_path))
    page = doc[0]                            # 첫 페이지만 사용
    mat = fitz.Matrix(2.0, 2.0)             # 2배 확대 → 해상도 향상 (OCR 정확도)
    pix = page.get_pixmap(matrix=mat)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pix.save(tmp.name)
    doc.close()
    return Path(tmp.name)


def image_to_base64(image_path: Path) -> tuple[str, str]:
    ext_to_mime = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
    }
    suffix = image_path.suffix.lower()
    media_type = ext_to_mime.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode("utf-8")

    return b64_str, media_type

# =============================================================================
# ── OCR + 항목 추출 (GPT-4o Vision)
# =============================================================================

def run_ocr_and_extract(client: OpenAI, image_path: Path) -> tuple[str, dict]:
    """
    GPT-4o Vision API를 호출해 이미지에서 텍스트 인식과 항목 추출을 동시에 수행한다.

    호출 흐름:
        1. 이미지를 Base64로 인코딩
        2. GPT-4o에 이미지 + 추출 프롬프트 전달
        3. JSON 응답 파싱 → 항목 딕셔너리 반환
        4. raw 텍스트(JSON 문자열)도 함께 반환해 UI에 표시

    Args:
        client (OpenAI): 초기화된 OpenAI 클라이언트
        image_path (Path): 처리할 이미지 경로

    Returns:
        tuple:
            - raw_text (str): GPT가 반환한 JSON 원문 (OCR 원문 역할)
            - fields (dict): 파싱된 항목 딕셔너리

    Raises:
        ValueError: JSON 파싱 실패 시
        openai.OpenAIError: API 호출 실패 시
    """
    b64_str, media_type = image_to_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",          # Vision 지원 모델
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    # ① 텍스트 지시
                    {"type": "text", "text": EXTRACT_PROMPT},
                    # ② 이미지 (Base64)
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_str}",
                            "detail": "high",  # 고해상도 모드: 문자 인식 정확도 향상
                        },
                    },
                ],
            }
        ],
    )

    raw_text = response.choices[0].message.content.strip()

    # JSON 파싱: 마크다운 펜스(```json ... ```)가 포함된 경우 제거
    clean_json = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        fields = json.loads(clean_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"GPT 응답을 JSON으로 파싱할 수 없습니다.\n응답 내용:\n{raw_text}") from e

    # FIELD_NAMES에 없는 키 제거, 누락된 키는 빈 문자열로 채움
    fields = {name: str(fields.get(name, "")).strip() for name in FIELD_NAMES}

    return raw_text, fields


# =============================================================================
# ── 표 ↔ 딕셔너리 변환 유틸리티
# =============================================================================

def fields_to_table(fields: dict) -> pd.DataFrame:
    """항목 딕셔너리 → 2열(항목, 값) DataFrame"""
    return pd.DataFrame([{"항목": name, "값": fields.get(name, "")} for name in FIELD_NAMES])


def table_to_fields(table: pd.DataFrame) -> dict:
    """편집된 DataFrame → 항목 딕셔너리"""
    cleaned = table.copy()
    cleaned["항목"] = cleaned["항목"].astype(str).str.strip()
    cleaned["값"]   = cleaned["값"].fillna("").astype(str).str.strip()
    return {row["항목"]: row["값"] for _, row in cleaned.iterrows() if row["항목"]}


# =============================================================================
# ── 파일 출력 유틸리티
# =============================================================================

def make_excel_bytes(fields: dict, raw_text: str) -> bytes:
    """
    추출 결과(추출결과 시트)와 GPT 원문 응답(OCR원문 시트)을
    2개 시트 Excel 파일로 메모리에 생성해 반환한다.
    """
    output = BytesIO()
    result_df = fields_to_table(fields)
    ocr_df    = pd.DataFrame({"GPT 응답 원문": raw_text.splitlines() or [""]})

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="추출결과")
        ocr_df.to_excel(writer,    index=False, sheet_name="OCR원문")

    return output.getvalue()


def save_json(fields: dict) -> Path:
    """수정된 결과를 타임스탬프 JSON 파일로 OUTPUT_DIR에 저장한다."""
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"business_license_result_{timestamp}.json"
    output_path.write_text(
        json.dumps(fields, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


# =============================================================================
# ── Streamlit UI
# =============================================================================

st.title("🏢 사업자등록증 OCR (경량버전)")
st.caption("EasyOCR 없이 OpenAI GPT-4o Vision API로 항목을 추출합니다.")

# ── 처리 단계 배지 ────────────────────────────────────────────────────────────
for col, label in zip(
    st.columns(6),
    ["① 업로드", "② 미리보기", "③ OCR 텍스트", "④ 추출 표", "⑤ 수정", "⑥ 저장/다운로드"],
):
    col.markdown(
        f"<div style='text-align:center;background:#f0f2f6;"
        f"border-radius:8px;padding:6px 2px;font-size:0.82rem;'>{label}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── 사이드바: API Key 입력 ────────────────────────────────────────────────────
# 기존 코드
#with st.sidebar:
#    st.header("⚙️ 설정")
#    api_key = st.text_input(
#        "OpenAI API Key",
#        type="password",                        # 입력값 마스킹
#        placeholder="sk-",
#        help="https://platform.openai.com/api-keys 에서 발급받으세요.",
#    )
#    st.caption("API Key는 이 세션 내에서만 사용되며 서버에 저장되지 않습니다.")
#    st.divider()
#    st.markdown("**필요 패키지**")
#    st.code("pip install streamlit openai pillow pandas openpyxl", language="bash")

# 변경 코드 - Secrets에서 자동으로 읽어옴
api_key = st.secrets["OPENAI_API_KEY"]

with st.sidebar:
    st.header("⚙️ 설정")
    st.success("API Key가 서버에 등록되어 있습니다.")

# ── API Key 미입력 시 안내 ────────────────────────────────────────────────────
if not api_key:
    st.warning("👈 왼쪽 사이드바에서 OpenAI API Key를 먼저 입력하세요.")
    st.stop()  # Key 없으면 이하 UI 렌더링 중단

# OpenAI 클라이언트 생성
client = get_client(api_key)

# =============================================================================
# ── 단계 1: 파일 업로드
# =============================================================================

st.subheader("1️⃣ 사업자등록증 업로드")
uploaded_file = st.file_uploader(
    "사업자등록증 이미지 파일 또는 PDF",
    type=["png", "jpg", "jpeg", "webp", "pdf"],  # PDF 추가
    help="PNG / JPG / JPEG / WEBP / PDF 형식 지원",
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(uploaded_file.name).suffix,
    ) as tmp:
        tmp.write(uploaded_file.getbuffer())
        upload_path = Path(tmp.name)

    # PDF면 첫 페이지를 이미지로 변환, 아니면 그대로 사용
    is_pdf = upload_path.suffix.lower() == ".pdf"
    image_path = pdf_to_image(upload_path) if is_pdf else upload_path

    # ── OCR 실행 ─────────────────────────────────────────────────────────────
    with st.spinner("🔍 GPT-4o Vision으로 분석 중입니다..."):
        try:
            raw_text, extracted_fields = run_ocr_and_extract(client, image_path)
        except ValueError as e:
            st.error(f"❌ 파싱 오류: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ API 호출 오류: {e}")
            st.stop()

    # =========================================================================
    # ── 단계 2 & 3: 이미지 미리보기 + GPT 응답 원문
    # =========================================================================

    preview_col, ocr_col = st.columns([1, 1], gap="large")

    with preview_col:
        st.subheader("2️⃣ 원본 이미지 미리보기")
        st.image(Image.open(image_path), use_container_width=True)

    with ocr_col:
        st.subheader("3️⃣ GPT 응답 원문 (OCR 결과)")
        st.caption("GPT-4o가 반환한 JSON 원문입니다.")
        st.text_area("GPT 응답", raw_text, height=350)

    st.divider()

    # =========================================================================
    # ── 단계 4: 추출 결과 표 (읽기 전용)
    # =========================================================================

    st.subheader("4️⃣ 추출 결과 표 보기")
    result_table = fields_to_table(extracted_fields)
    st.dataframe(result_table, hide_index=True, use_container_width=True)

    st.divider()

    # =========================================================================
    # ── 단계 5: 사용자 직접 수정
    # =========================================================================

    st.subheader("5️⃣ 사용자가 직접 수정")
    st.caption("값 셀을 클릭해 수정할 수 있습니다. 항목명은 수정되지 않습니다.")

    edited_table = st.data_editor(
        result_table,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "항목": st.column_config.TextColumn("항목", disabled=True),
            "값":   st.column_config.TextColumn("값"),
        },
        key=f"editor_{uploaded_file.name}_{uploaded_file.size}",
    )
    edited_fields = table_to_fields(edited_table)

    st.divider()

    # =========================================================================
    # ── 단계 6: 저장 / 다운로드
    # =========================================================================

    st.subheader("6️⃣ 저장 / 다운로드")

    save_col, json_col, excel_col = st.columns(3, gap="medium")

    with save_col:
        st.markdown("**📁 서버 저장**")
        if st.button("수정 결과 저장 (서버)", use_container_width=True):
            saved_path = save_json(edited_fields)
            st.success(f"✅ 저장 완료: `{saved_path.name}`")

    with json_col:
        st.markdown("**⬇️ JSON 다운로드**")
        st.download_button(
            label="JSON 다운로드",
            data=json.dumps(edited_fields, ensure_ascii=False, indent=2),
            file_name="business_license_result.json",
            mime="application/json",
            use_container_width=True,
        )

    with excel_col:
        st.markdown("**📊 엑셀 다운로드**")
        st.download_button(
            label="엑셀 다운로드 (.xlsx)",
            data=make_excel_bytes(edited_fields, raw_text),
            file_name="business_license_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with st.expander("📋 최종 JSON 미리보기", expanded=False):
        st.json(edited_fields)

else:
    st.info(
        "👆 이미지를 업로드하면 ①~⑥ 단계가 표시됩니다.\n\n"
        "- **개인정보 주의**: 이미지가 OpenAI 서버로 전송됩니다. 실제 문서는 마스킹 후 사용하세요.\n"
        "- **API 비용**: GPT-4o Vision 호출 1회당 약 $0.01~0.03 수준입니다."
    )
