from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt

ann_file = "./data/coco/annotations/instances_val2017.json"
coco = COCO(ann_file)

# 1. Lấy image
img_id = 568195
# coco.getImgIds()[0]

# 2. Lấy category id của person
person_cat_id = coco.getCatIds(catNms=["person"])[0]

# 3. Lấy annotation của person trong ảnh
ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_cat_id])
anns = coco.loadAnns(ann_ids)
print(img_id)

img_info = coco.loadImgs(img_id)[0]
print(f"Kích thước ảnh từ COCO: Height={img_info['height']}, Width={img_info['width']}")

# 4. Gộp tất cả person mask (nếu có nhiều người)
img_info = coco.loadImgs(img_id)[0]
H, W = img_info['height'], img_info['width']
person_mask = np.zeros((H, W), dtype=np.uint8)

for ann in anns:
    person_mask |= coco.annToMask(ann)
print(person_mask)
plt.imshow(person_mask, cmap="gray")
plt.title("Person Mask")
plt.axis("off")
plt.savefig("person_mask.png", dpi=150)
print("Saved person_mask.png")
print(img_id)


# from pycocotools.coco import COCO

# ann_file = "./data/coco/annotations/stuff_val2017.json"
# coco = COCO(ann_file)

# cat_ids = coco.getCatIds()
# cats = coco.loadCats(cat_ids)

# print("Số lượng class trong COCO-Stuff:", len(cats))

# # In thử vài class đầu
# for c in cats:
#     print(c["id"], c["name"])
