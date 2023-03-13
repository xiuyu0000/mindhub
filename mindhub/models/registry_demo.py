
if __name__ == "__main__":
    from mindhub.models.registry import load_local_model, local_models, list_models_info
    # py_model = load_local_model("TinyDarkNetImageNet", "/data1/hjs/mindhub/resource/tinydarknet_imagenet")
    load_local_model("TinyDarkNetImageNet", "/data1/hjs/mindhub/resource/tinydarknet_imagenet")
    model = local_models("tinydarknet_imagenet")
    # print(py_model("tinydarknet_imagenet",pretrained=True))
    print(list_models_info("darknet", True))
    print(model("tinydarknet_imagenet", True))
