
if __name__ == "__main__":
    import os
    from mindhub.models.model import Model

    net = Model("tinydarknet_imagenet", "/data1/hjs/mindhub/resource/tinydarknet_imagenet/", pretrained=True)
    print(net.infer("/data1/tinydarknet/data/infer/n02090622/",
                    os.path.join("/data1/hjs/mindhub/resource/tinydarknet_imagenet/",
                                 "./label_map.json")))
