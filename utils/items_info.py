#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""商品目录、名称解析和价格信息。"""


# P01-P06 是稳定的商品/YOLO 类别；A1-A4、B1-B4 只表示货架位置。
PRODUCTS_BY_CODE = {
    "P01": {
        "name": "外星人电解质水",
        "aliases": ("外星人", "外星人水", "电解质水"),
        "total_height_cm": 21.5,
        "graspable_height_cm": 7.5,
        "conservative_max_width_cm": 6.8,
    },
    "P02": {
        "name": "维他命水",
        "aliases": ("维他命",),
        "total_height_cm": 19.0,
        "graspable_height_cm": 5.0,
        "conservative_max_width_cm": 6.5,
    },
    "P03": {
        "name": "统一阿萨姆原味奶茶",
        "aliases": ("阿萨姆", "阿萨姆奶茶", "统一阿萨姆", "奶茶"),
        "total_height_cm": 22.3,
        "graspable_height_cm": 17.5,
        "conservative_max_width_cm": 6.0,
    },
    "P04": {
        "name": "维他鸭屎香柠檬茶",
        "aliases": ("维他柠檬茶", "鸭屎香", "鸭屎香柠檬茶", "维他鸭屎香"),
        "total_height_cm": 21.5,
        "graspable_height_cm": 17.5,
        "conservative_max_width_cm": 7.3,
    },
    "P05": {
        "name": "百事可乐",
        "aliases": ("百事", "可乐"),
        "total_height_cm": 24.5,
        "graspable_height_cm": 20.0,
        "conservative_max_width_cm": 7.0,
    },
    "P06": {
        "name": "美年达橙味汽水",
        "aliases": (
            "美年达",
            "美年达橙味",
            "橙子汽水",
            "橙汁汽水",
            "橙味汽水",
        ),
        "total_height_cm": 25.5,
        "graspable_height_cm": 20.0,
        "conservative_max_width_cm": 7.0,
    },
}


# 暂时保留旧价格表；没有提供价格的新商品仍返回 0.0。
ITEM_PRICES = {
    "可口可乐": 3.5,
    "百事可乐": 3.5,
    "雪碧": 3.5,
    "红牛": 6.0,
    "矿泉水": 2.5,
    "营养快线": 6.0,
    "AD钙奶": 5.5,
    "纯牛奶": 2.5,
    "雀巢咖啡": 4.0,
    "牙膏": 7.0,
    "洗发水": 12.0,
    "薯片": 3.5,
    "奥利奥饼干": 6.0,
    "纸巾": 4.0,
    "橘子": 1.5,
    "苹果": 2.0,
}


def _normalize_item_name(value: str) -> str:
    return "".join(value.split()).casefold()


def _build_alias_index(products: dict) -> dict[str, str]:
    index: dict[str, str] = {}
    for product_code, item in products.items():
        height = float(item["total_height_cm"])
        graspable_height = float(item["graspable_height_cm"])
        max_width = float(item["conservative_max_width_cm"])
        if not 0 < graspable_height <= height or max_width <= 0:
            raise ValueError(f"商品 {product_code} 的尺寸无效")
        for name in (product_code, item["name"], *item["aliases"]):
            key = _normalize_item_name(name)
            previous = index.get(key)
            if previous is not None and previous != product_code:
                raise ValueError(
                    f"商品名称或简称有歧义: {name!r} 同时属于 {previous} 和 {product_code}"
                )
            index[key] = product_code
    return index


_PRODUCT_CODE_BY_NAME = _build_alias_index(PRODUCTS_BY_CODE)


def resolve_product_code(name: str) -> str | None:
    """把商品编号、正式名称或显式简称解析成唯一的 P01-P06。"""
    return _PRODUCT_CODE_BY_NAME.get(_normalize_item_name(name))


def get_item_info(name: str) -> dict | None:
    """返回商品信息；不做容易误选商品的模糊/包含匹配。"""
    product_code = resolve_product_code(name)
    if product_code is None:
        return None
    return {"product_code": product_code, **PRODUCTS_BY_CODE[product_code]}


def get_item_price(item_name: str) -> float:
    """获取商品价格。"""
    item = get_item_info(item_name)
    canonical_name = item["name"] if item is not None else item_name
    return ITEM_PRICES.get(canonical_name, 0.0)


def get_all_items_with_prices() -> dict:
    """获取所有商品及其价格。"""
    return ITEM_PRICES.copy()


def get_all_supported_items() -> list:
    """获取当前机器人支持的商品名称。"""
    return [item["name"] for item in PRODUCTS_BY_CODE.values()]
