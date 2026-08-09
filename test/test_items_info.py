"""Checks for the current six-product catalog."""

from utils import items_info


def test_product_dimensions_and_aliases_are_unambiguous():
    expected = {
        "P01": ("外星人电解质水", 21.5, 7.5, 6.8),
        "P02": ("维他命水", 19.0, 5.0, 6.5),
        "P03": ("统一阿萨姆原味奶茶", 22.3, 17.5, 6.0),
        "P04": ("维他鸭屎香柠檬茶", 21.5, 17.5, 7.3),
        "P05": ("百事可乐", 24.5, 20.0, 7.0),
        "P06": ("美年达橙味汽水", 25.5, 20.0, 7.0),
    }

    for code, (name, height, graspable_height, max_width) in expected.items():
        item = items_info.get_item_info(code)
        assert item["name"] == name
        assert item["total_height_cm"] == height
        assert item["graspable_height_cm"] == graspable_height
        assert item["conservative_max_width_cm"] == max_width
        assert items_info.resolve_product_code(name) == code
        for alias in item["aliases"]:
            assert items_info.resolve_product_code(alias) == code

    assert items_info.resolve_product_code("  百事  ") == "P05"
    assert items_info.resolve_product_code("电解质水") == "P01"
    assert items_info.resolve_product_code("奶茶") == "P03"
    assert items_info.resolve_product_code("维他柠檬茶") == "P04"
    assert items_info.resolve_product_code("可乐") == "P05"
    assert items_info.resolve_product_code("橙子汽水") == "P06"
    assert items_info.resolve_product_code("橙汁汽水") == "P06"
    assert items_info.resolve_product_code("橙味汽水") == "P06"
    assert items_info.resolve_product_code("维他") is None
