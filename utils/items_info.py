#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品信息数据库
包含商品价格信息
注意：YOLO模型应配置为直接输出中文标签，无需代码转换
"""

# 商品价格映射表（YOLO模型配置为使用中文标签）
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
    "苹果": 2.0
}

def get_item_price(item_name: str) -> float:
    """获取商品价格"""
    return ITEM_PRICES.get(item_name, 0.0)

def get_all_items_with_prices() -> dict:
    """获取所有商品及其价格"""
    return ITEM_PRICES.copy()

def get_all_supported_items() -> list:
    """获取所有支持的商品列表"""
    return list(ITEM_PRICES.keys())