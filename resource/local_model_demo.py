
if __name__ == "__main__":

    import importlib.util
    import sys

    sys.path.insert(0, "/data1/hjs/mindhub/resource/tinydarknet_imagenet/")

    # Load the module from the specified file path
    spec = importlib.util.spec_from_file_location("TinyDarkNetImageNet", "/data1/hjs/mindhub/resource/tinydarknet_imagenet/mindspore_hub_conf.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    model = model_module.CrossEntropySmooth()

    print(model)

    sys.path.pop(0)
