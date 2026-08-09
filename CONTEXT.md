# Shelf Dispensing

This context names the products and shelf locations used by the dispensing robot.

## Language

**Product code**:
A stable `P01`-style identity for one product and its future YOLO class.
_Avoid_: Letter, shelf code

**Shelf slot**:
One physical storage position: `A1`-`A4` on the upper layer or `B1`-`B4` on the lower layer.
_Avoid_: Product code, point

**Canonical product name**:
The single full Chinese name displayed for a product.
_Avoid_: YOLO name, nickname

**Product alias**:
An explicitly approved, full-string alternative that identifies exactly one product in the current supported catalog; a generic drink name is valid while it remains unique there.
_Avoid_: Fuzzy name, keyword

**Total height**:
The measured end-to-end height of a product container.
_Avoid_: Graspable height

**Graspable height**:
The vertical length of the relatively wide container region suitable for grasping.
_Avoid_: Grasp height, total height

**Conservative maximum width**:
An approximate widest container dimension that may include a small allowance.
_Avoid_: Exact diameter
