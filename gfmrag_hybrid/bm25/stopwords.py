"""
Stopwords cho nhánh truy hồi BM25.

Các tập stopword được truyền vào ``BM25Searcher`` qua constructor; mỗi script
inference chọn tập phù hợp với ngôn ngữ corpus của mình (vi/en).
"""

VIETNAMESE_STOPWORDS = {
    "và", "là", "của", "có", "trong", "về", "cách", "nó", "thể", "hỗ", "trợ",
    "trường", "hợp", "do", "quá", "liều", "các", "những", "cho", "để", "với",
    "không", "khi", "được", "một", "này", "đó", "thuốc", "bệnh", "sự", "bị",
    "ra", "vào", "tôi", "cần", "tìm", "thêm", "thông", "tin", "chi", "tiết",
    "làm", "sao", "như", "thế", "nào", "thực", "thể", "thiếu", "tương", "tác"
}

ENGLISH_STOPWORDS = {
    "and", "is", "of", "in", "to", "the", "a", "an", "for", "with", "on", "as",
    "by", "at", "from", "or", "that", "this", "it", "are", "was", "were", "be",
    "has", "have", "had", "not", "but", "what", "how", "when", "where", "who"
}
