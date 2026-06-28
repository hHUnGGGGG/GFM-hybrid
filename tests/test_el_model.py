def test_colbert_el_model() -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag_hybrid.kg_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": "colbert-ir/colbertv2.0",
            "root": "tmp",
            "force": False,
        }
    )

    el_model = instantiate(cfg)

    # Entity từ query y tế tiếng Việt
    ner_entity_list = ["Augmentin", "dị ứng Penicillin"]

    # Entity list từ Medical KG VN
    entity_list = [
        "Augmentin",
        "amoxicillin",
        "acid clavulanic",
        "Beta-lactam",
        "Penicillin",
        "dị ứng Penicillin",
        "dị ứng chéo Beta-lactam",
        "sốc phản vệ",
        "nổi mề đay",
        "nhiễm khuẩn đường hô hấp",
        "nhiễm khuẩn tiết niệu",
        "nhiễm khuẩn da",
        "viêm phổi cộng đồng",
        "viêm phế quản",
        "Azithromycin",
        "Macrolide",
        "Levofloxacin",
        "Fluoroquinolone",
        "625mg",
        "500mg",
        "suy gan nặng",
        "chống chỉ định",
        "tác dụng phụ",
        "liều dùng người lớn",
        "Metformin",
        "tiểu đường type 2",
        "Omeprazole",
        "loét dạ dày tá tràng",
        "tăng tiết acid",
        "Dexacin",
        "corticoid",
        "hội chứng Cushing",
        "suy thượng thận",
        "Biviantac",
        "đau vùng thượng vị",
        "trướng bụng",
        "ợ nóng",
    ]

    el_model.index(entity_list)
    linked_entity_dict = el_model(ner_entity_list, topk=3)
    print(linked_entity_dict)
    assert isinstance(linked_entity_dict, dict)

    # Kiểm tra kết quả có hợp lý không
    print("\n=== Kiểm tra kết quả ===")
    for query_entity, matches in linked_entity_dict.items():
        print(f"\nQuery entity: '{query_entity}'")
        for match in matches:
            print(f"  → {match['entity']} (score: {match['score']:.4f}, norm: {match['norm_score']:.4f})")


def test_dpr_el_model() -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag_hybrid.kg_construction.entity_linking_model.DPRELModel",
            "model_name": "intfloat/multilingual-e5-base",  # đổi sang multilingual
            "root": "tmp",
            "use_cache": True,
            "normalize": True,
            "batch_size": 32,
            "sim_batch_size": 512,
            "chunk_size": 100,
            "query_instruct": "query: ",       # prefix cho E5 model
            "passage_instruct": "passage: ",   # prefix cho E5 model
        }
    )

    el_model = instantiate(cfg)

    # Entity từ query y tế tiếng Việt
    # Test cả trường hợp: đúng chính xác, viết tắt, tên thương mại vs hoạt chất
    ner_entity_list = [
        "Augmentin",           # tên thương mại → phải match "Augmentin" hoặc "amoxicillin"
        "dị ứng Penicillin",   # đúng chính xác → phải match "dị ứng Penicillin"
        "DM type 2",           # viết tắt → phải match "tiểu đường type 2"
        "OPZ",                 # viết tắt → phải match "Omeprazole"
    ]

    # Entity list từ Medical KG VN
    entity_list = [
        "Augmentin",
        "amoxicillin",
        "acid clavulanic",
        "Beta-lactam",
        "Penicillin",
        "dị ứng Penicillin",
        "dị ứng chéo Beta-lactam",
        "sốc phản vệ",
        "nổi mề đay",
        "nhiễm khuẩn đường hô hấp",
        "nhiễm khuẩn tiết niệu",
        "nhiễm khuẩn da",
        "viêm phổi cộng đồng",
        "viêm phế quản",
        "Azithromycin",
        "Macrolide",
        "Levofloxacin",
        "Fluoroquinolone",
        "625mg",
        "500mg",
        "suy gan nặng",
        "chống chỉ định",
        "tác dụng phụ",
        "liều dùng người lớn",
        "Metformin",
        "tiểu đường type 2",
        "Omeprazole",
        "loét dạ dày tá tràng",
        "tăng tiết acid",
        "Dexacin",
        "corticoid",
        "hội chứng Cushing",
        "suy thượng thận",
        "Biviantac",
        "đau vùng thượng vị",
        "trướng bụng",
        "ợ nóng",
    ]

    el_model.index(entity_list)
    linked_entity_dict = el_model(ner_entity_list, topk=3)
    print(linked_entity_dict)
    assert isinstance(linked_entity_dict, dict)

    # Kiểm tra chất lượng entity linking
    print("\n=== Kiểm tra chất lượng Entity Linking ===")
    expected = {
        "Augmentin": "Augmentin",
        "dị ứng Penicillin": "dị ứng Penicillin",
        "DM type 2": "tiểu đường type 2",
        "OPZ": "Omeprazole",
    }
    passed = 0
    for query_entity, matches in linked_entity_dict.items():
        top1 = matches[0]["entity"]
        expected_entity = expected.get(query_entity, "?")
        ok = "✅" if top1 == expected_entity else "❌"
        print(f"{ok} '{query_entity}' → '{top1}' (expected: '{expected_entity}')")
        if top1 == expected_entity:
            passed += 1

    print(f"\nKết quả: {passed}/{len(expected)} entity linked đúng")
    print("Nếu DM type 2 và OPZ không match → multilingual-e5 cần fine-tune thêm")


if __name__ == "__main__":
    # Chạy DPR trước vì hỗ trợ tiếng Việt tốt hơn ColBERT
    test_dpr_el_model()
    # test_colbert_el_model()  # ColBERT không hỗ trợ tốt tiếng Việt