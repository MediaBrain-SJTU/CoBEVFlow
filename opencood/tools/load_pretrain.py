def rename_model_dict_keys(pretrained_dict_path, rename_dict):
    """ load pretrained state dict, keys may not match with model

    Args:
        model: nn.Module

        pretrained_dict: collections.OrderedDict

    """
    pretrained_dict = torch.load(pretrained_dict_path)
    # 1. filter out unnecessary keys
    for oldname, newname in rename_dict.items():
        if oldname.endswith("*"):
            _oldnames = list(pretrained_dict.keys())
            _oldnames = [x for x in _oldnames if x.startswith(oldname[:-1])]
            for _oldname in _oldnames:
                if newname != "":
                    _newname = _oldname.replace(oldname[:-1], newname[:-1])
                    pretrained_dict[_newname] = pretrained_dict[_oldname]
                pretrained_dict.pop(_oldname)
        else:
            if newname != "":
                pretrained_dict[newname] = pretrained_dict[oldname]
            pretrained_dict.pop(oldname)
    torch.save(pretrained_dict, pretrained_dict_path)


if __name__ == "__main__":
    dict_path = "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs/then_SECOND_LID_72_48/net_epoch1.pth"
    rename_dict = {"lidar_encoder.*": "",
                   "shrink_lidar.*": ""}
    rename_model_dict_keys(dict_path, rename_dict)

    rename_dict = {"lidar_encoder.*": "", 
                "shrink_lidar.*": ""} # if value is empty, pop this key, if value is non empty, replace this key with new name. * means all keys start with this prefix