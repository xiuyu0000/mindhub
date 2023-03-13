
if __name__ == "__main__":
    from mindhub.models.model import Model
    model = Model("tinydarknet_imagenet", "/data1/hjs/mindhub_repo/resource/tinydarknet_imagenet", pretrained=True)
    # model = Model("tinydarknet_imagenet", pretrained=True)
