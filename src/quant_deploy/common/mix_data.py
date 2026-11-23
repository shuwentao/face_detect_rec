import json


def merge_extra(ds, ann_json, img_dir, repeat=1):
    """把额外数据集并入 ds；repeat 提升其采样权重。"""
    ex = json.load(open(ann_json))
    base_i = max(i["id"] for i in ds.images) + 1
    all_a = [a for v in ds.by_image.values() for a in v]
    base_a = max((a["id"] for a in all_a), default=0) + 1
    by_ex = {}
    for a in ex["annotations"]:
        by_ex.setdefault(a["image_id"], []).append(a)

    n_img = n_ann = 0
    for _r in range(repeat):
        for info in ex["images"]:
            nid = base_i + n_img
            new = dict(info)
            new["id"] = nid
            new["_img_dir"] = img_dir
            ds.images.append(new)
            n_img += 1
            olds = by_ex.get(info["id"], [])
            ns = []
            for a in olds:
                b = dict(a)
                b["id"] = base_a + n_ann
                b["image_id"] = nid
                ns.append(b)
                n_ann += 1
            ds.by_image[nid] = ns
    return n_img, n_ann
