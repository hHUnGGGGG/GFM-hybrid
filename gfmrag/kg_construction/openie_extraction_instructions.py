from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# ============================================================
# ONE-SHOT EXAMPLE
# ============================================================

one_shot_passage = """
Augmentin là kháng sinh phối hợp giữa amoxicillin và acid clavulanic, thuộc nhóm Beta-lactam.
Sản xuất bởi GlaxoSmithKline. Dạng bào chế: viên nén bao phim.
Thuốc được chỉ định điều trị các bệnh nhiễm khuẩn đường hô hấp, tiết niệu và da.
Cơ chế tác dụng: amoxicillin ức chế tổng hợp vách tế bào vi khuẩn.
Augmentin có thể gây tiêu chảy (thường gặp), vàng da ứ mật (hiếm gặp).
Augmentin chống chỉ định với bệnh nhân tiền sử vàng da/rối loạn gan do amoxicillin/clavulanate.
Thận trọng với phụ nữ cho con bú. Tương tác với warfarin, methotrexat.
Liều dùng người lớn: 625mg, uống 2-3 lần mỗi ngày trong 7-10 ngày, uống trong bữa ăn.
Bảo quản nơi khô ráo, nhiệt độ không quá 30°C. Hạn sử dụng: 24 tháng.
"""

one_shot_passage_entities = """
{
 "named_entities": [
  "augmentin",
  "amoxicillin",
  "acid clavulanic",
  "beta-lactam",
  "glaxosmithkline",
  "viên nén bao phim",
  "nhiễm khuẩn đường hô hấp",
  "nhiễm khuẩn tiết niệu",
  "nhiễm khuẩn da",
  "ức chế tổng hợp vách tế bào vi khuẩn",
  "tiêu chảy (thường gặp)",
  "vàng da ứ mật (hiếm gặp)",
  "tiền sử vàng da",
  "rối loạn gan do amoxicillin/clavulanate",
  "phụ nữ cho con bú",
  "warfarin",
  "methotrexat",
  "625mg 2-3 lần/ngày trong 7-10 ngày (người lớn)",
  "đường uống",
  "uống trong bữa ăn",
  "nơi khô ráo, không quá 30°c",
  "24 tháng"
 ]
}
"""

one_shot_passage_triples = """
{
 "triples": [
  ["augmentin", "thành phần", "amoxicillin"],
  ["augmentin", "thành phần", "acid clavulanic"],
  ["augmentin", "thuộc nhóm", "beta-lactam"],
  ["augmentin", "sản xuất bởi", "glaxosmithkline"],
  ["augmentin", "dạng bào chế", "viên nén bao phim"],
  ["augmentin", "điều trị", "nhiễm khuẩn đường hô hấp"],
  ["augmentin", "điều trị", "nhiễm khuẩn tiết niệu"],
  ["augmentin", "điều trị", "nhiễm khuẩn da"],
  ["amoxicillin", "cơ chế tác dụng", "ức chế tổng hợp vách tế bào vi khuẩn"],
  ["augmentin", "gây ra", "tiêu chảy (thường gặp)"],
  ["augmentin", "gây ra", "vàng da ứ mật (hiếm gặp)"],
  ["augmentin", "chống chỉ định", "tiền sử vàng da"],
  ["augmentin", "chống chỉ định", "rối loạn gan do amoxicillin/clavulanate"],
  ["augmentin", "thận trọng với", "phụ nữ cho con bú"],
  ["augmentin", "tương tác với", "warfarin"],
  ["augmentin", "tương tác với", "methotrexat"],
  ["augmentin", "liều dùng", "625mg 2-3 lần/ngày trong 7-10 ngày (người lớn)"],
  ["augmentin", "đường dùng", "đường uống"],
  ["augmentin", "lưu ý sử dụng", "uống trong bữa ăn"],
  ["augmentin", "bảo quản", "nơi khô ráo, không quá 30°c"],
  ["augmentin", "hạn sử dụng", "24 tháng"]
 ]
}
"""

# ============================================================
# NER PROMPT
# ============================================================

ner_instruction = """
Bạn là hệ thống trích xuất thực thể y khoa tiếng Việt.

Nhiệm vụ:
Trích xuất các thực thể y tế từ đoạn văn theo đúng các loại sau:

DRUG        : tên thuốc thương mại, hoạt chất (vd: Augmentin, amoxicillin)
DRUG_CLASS  : nhóm dược lý (vd: beta-lactam, kháng viêm không steroid)
DISEASE     : bệnh lý, chẩn đoán (vd: tăng huyết áp, loét dạ dày)
SYMPTOM     : triệu chứng, tác dụng phụ (vd: tiêu chảy, phù mạch)
DOSAGE      : liều lượng, tần suất, thời gian dùng (vd: 625mg, 2-3 lần/ngày)
ANATOMY     : cơ quan cơ thể (vd: gan, thận)
MANUFACTURER: tên công ty sản xuất
FORMULATION : dạng bào chế (vd: viên nén bao phim, siro)
EXCIPIENT   : tá dược (vd: lactose monohydrat, magnesi stearat)
PATIENT_GROUP: nhóm bệnh nhân đặc biệt (vd: phụ nữ có thai, trẻ em dưới 12 tuổi)
STORAGE     : điều kiện bảo quản, hạn sử dụng
              (vd: dưới 30°c, nơi khô ráo, tránh ánh sáng, 36 tháng)
INSTRUCTION : hướng dẫn cách dùng, thời điểm dùng, xử trí sự cố
              (vd: uống sau bữa ăn, không dùng gấp đôi liều,
                   rửa dạ dày, viên thuốc bị biến đổi màu)

Quy tắc:
- Viết thường toàn bộ entity
- Giữ nguyên từ trong tài liệu, không paraphrase
- Entity ngắn gọn, chỉ dài khi rút gọn mất nghĩa
- Không trích xuất: động từ đơn lẻ, giới từ, liên từ

Chỉ trả về JSON — KHÔNG giải thích, KHÔNG thêm text ngoài JSON:
{"named_entities": []}
"""

ner_input_one_shot = f"""
Đoạn văn:

{one_shot_passage}

"""

ner_user_input = """
Đoạn văn:

{user_input}

"""

ner_prompts = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=ner_instruction),
        HumanMessage(content=ner_input_one_shot),
        AIMessage(content=one_shot_passage_entities),
        HumanMessagePromptTemplate.from_template(ner_user_input),
    ]
)


# ============================================================
# OPENIE PROMPT
# ============================================================

openie_post_ner_instruction = """
Bạn là hệ thống xây dựng Knowledge Graph dược phẩm tiếng Việt.

Nhiệm vụ:
Tạo các triple (subject, relation, object) từ đoạn văn, dựa trên danh sách thực thể đã cho.

BẮT BUỘC:
- Chỉ trả về JSON, không giải thích, không thêm text ngoài JSON
- JSON phải có key "triples"
- Mỗi triple là ["subject", "relation", "object"]
- Tất cả viết thường
- Nếu đoạn văn có thông tin y tế, PHẢI tạo ít nhất 5 triple
- KHÔNG BAO GIỜ trả về {"triples": []}

ĐỊNH DẠNG:
{
 "triples": [
  ["entity1", "relation", "entity2"]
 ]
}

═══════════════════════════════════════
RELATION SCHEMA — CHỈ DÙNG 20 RELATION NÀY
═══════════════════════════════════════

NHÓM 1 — THÔNG TIN CƠ BẢN
  đồng nghĩa      → tên gọi khác của cùng một thuốc/hoạt chất
                    VD: ["adrenalin", "đồng nghĩa", "epinephrin"]
  thành phần      → hoạt chất hoặc tá dược có trong thuốc
                    VD: ["augmentin", "thành phần", "amoxicillin"]
  thuộc nhóm      → nhóm dược lý
                    VD: ["aspirin", "thuộc nhóm", "kháng viêm không steroid"]
  dạng bào chế    → hình thức vật lý của thuốc
                    VD: ["aspirin 81", "dạng bào chế", "viên bao phim tan trong ruột"]
  sản xuất bởi    → công ty sản xuất
                    VD: ["cefurich", "sản xuất bởi", "công ty cổ phần us pharma usa"]

NHÓM 2 — CHỈ ĐỊNH & CÁCH DÙNG
  điều trị        → bệnh lý hoặc triệu chứng được chỉ định
                    VD: ["captopril", "điều trị", "tăng huyết áp"]
  đường dùng      → cách đưa thuốc vào cơ thể (uống, tiêm, bôi...)
                    VD: ["crotamiton", "đường dùng", "bôi ngoài da"]
                    KHÔNG dùng cho thời điểm uống (trước/sau ăn) → dùng "lưu ý sử dụng"
  đối tượng sử dụng → nhóm tuổi/người được chỉ định liều riêng
                    VD: ["aspirin", "đối tượng sử dụng", "trẻ em trên 12 tuổi"]
  liều dùng       → lượng thuốc, tần suất, thời gian; ghi rõ đối tượng nếu có
                    VD: ["cefadroxil", "liều dùng", "1g/ngày uống 1 lần (người lớn)"]
  cơ chế tác dụng → cách thức thuốc hoạt động trong cơ thể
                    VD: ["captopril", "cơ chế tác dụng", "ức chế hệ renin-angiotensin-aldosteron"]

NHÓM 3 — CẢNH BÁO & AN TOÀN
  chống chỉ định  → tuyệt đối không được dùng; giữ nguyên điều kiện nếu có
                    VD: ["acnotin", "chống chỉ định", "phụ nữ có thai"]
  thận trọng với  → cần theo dõi sát, có thể dùng nhưng phải cẩn thận
                    VD: ["cefalexin", "thận trọng với", "suy thận"]
  tương tác với   → thuốc/thực phẩm làm thay đổi tác dụng; dùng INN, không dùng tên thương mại
                    VD: ["amoxicillin", "tương tác với", "thuốc tránh thai đường uống"]
  gây ra          → tác dụng phụ; ghi tần suất vào object nếu document phân tầng
                    VD: ["ibuprofen", "gây ra", "xuất huyết tiêu hóa (thường gặp)"]
                    VD: ["ceritine", "gây ra", "viêm gan (hiếm gặp)"]

NHÓM 4 — XỬ LÝ SỰ CỐ
  lưu ý sử dụng   → hướng dẫn hành vi khi dùng thuốc (thời điểm uống, cách uống, quên liều...)
                    VD: ["captopril", "lưu ý sử dụng", "uống trước ăn 1 giờ hoặc 2 giờ sau bữa ăn"]
                    VD: ["augmentin", "lưu ý sử dụng", "không dùng gấp đôi liều"]
  xử trí quá liều → biện pháp y tế khi dùng quá liều
                    VD: ["paracetamol", "xử trí quá liều", "rửa dạ dày"]
  ngưng dùng khi  → dấu hiệu cơ thể buộc phải dừng thuốc ngay
                    VD: ["asevictoria", "ngưng dùng khi", "chậm kinh"]
  không sử dụng khi → trạng thái vật lý thuốc bị hỏng
                    VD: ["ampicillin", "không sử dụng khi", "viên thuốc bị biến đổi màu"]

NHÓM 5 — BẢO QUẢN
  bảo quản        → điều kiện môi trường cất giữ thuốc
                    VD: ["bifril", "bảo quản", "dưới 30°c"]
  hạn sử dụng     → thời gian lưu trữ tối đa
                    VD: ["asevictoria", "hạn sử dụng", "36 tháng"]

═══════════════════════════════════════
QUY TẮC BẮT BUỘC
═══════════════════════════════════════

1. SUBJECT phải nằm trong named_entities
2. OBJECT phải nằm trong named_entities
3. Entity ngắn gọn — chỉ dài khi việc rút gọn làm mất nghĩa
4. Giải quyết đại từ: "thuốc này", "nó" → thay bằng tên thuốc cụ thể

5. TÁCH đường dùng và thời điểm dùng:
   ĐÚNG: ["captopril", "đường dùng", "đường uống"]
   ĐÚNG: ["captopril", "lưu ý sử dụng", "uống trước ăn 1 giờ hoặc 2 giờ sau bữa ăn"]
   SAI:  ["captopril", "đường dùng", "uống trước ăn 1 giờ"]

6. TÁCH thành phần và hàm lượng:
   ĐÚNG: ["ceritine", "thành phần", "cetirizine dihydrochloride"]
   SAI:  ["ceritine", "thành phần", "10mg"]

7. GHI RÕ đối tượng trong liều dùng khi có nhiều mức:
   ĐÚNG: ["biviantac", "liều dùng", "10g (1 gói) x 2-4 lần/ngày (người lớn)"]
   ĐÚNG: ["biviantac", "liều dùng", "5-10g x 2-4 lần/ngày (trẻ em)"]

8. NODE TRUNG GIAN cho hoạt chất có đặc tính riêng:
   ["augmentin", "thành phần", "amoxicillin"]
   ["amoxicillin", "cơ chế tác dụng", "ức chế tổng hợp vách tế bào vi khuẩn"]
   ["amoxicillin", "thuộc nhóm", "beta-lactam"]

9. KHÔNG TẠO TRIPLE VÔ NGHĨA:
   SAI: ["biviantac", "đường dùng", "lúc đói hoặc sau khi ăn 20 phút"]
   SAI: ["augmentin", "thành phần", "625mg"]
   SAI: ["ceritine", "thành phần", "10mg"]
   SAI: ["ampicillin", "gây ra", "miệng"]

10. NGHIÊM CẤM dùng relation ngoài 20 relations đã liệt kê.
    Mapping bắt buộc khi gặp relation không có trong schema:
    "có thể gây ra"    → "gây ra"
    "quá liều"         → "xử trí quá liều"
    "điều trị tại chỗ" → "điều trị"
    "liệu pháp hỗ trợ" → "điều trị"
    "sử dụng"          → "điều trị" hoặc "lưu ý sử dụng"
    "nguy cơ"          → "thận trọng với"
    "phối hợp thận trọng với" → "thận trọng với"
    "tăng tác dụng của" → "tương tác với"
    "làm giảm hiệu quả" → "tương tác với"
    "dự phòng"         → "điều trị"
    "không dùng cho"   → "chống chỉ định"
    "bài xuất vào"     → "thận trọng với"
"""

openie_post_ner_frame = """
Chuyển đoạn văn sau thành triple Knowledge Graph.

Đoạn văn:

{passage}


Danh sách thực thể:

{named_entity_json}
"""

openie_post_ner_input_one_shot = openie_post_ner_frame.replace(
    "{passage}", one_shot_passage
).replace("{named_entity_json}", one_shot_passage_entities)

openie_post_ner_prompts = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=openie_post_ner_instruction),
        HumanMessage(content=openie_post_ner_input_one_shot),
        AIMessage(content=one_shot_passage_triples),
        HumanMessagePromptTemplate.from_template(openie_post_ner_frame),
    ]
)