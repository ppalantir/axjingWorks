def _init():
    global _global_dict
    print("--- globalvar init --- \n")
    _global_dict = {}   #创建一个空字典
    _global_dict["workdir"] = "./"
    _global_dict["swim_split"] = "cut_data/"
    _global_dict["swim_raw"] = "uniformed_data/"
    _global_dict["swim_label"] = "./out/labelData/"

def set_value(name, value):
    _global_dict[name] = value

def get_value(name, defValue=None):
    try:
        return _global_dict[name]   #文件异常处理
    except KeyError:
        return defValue

if __name__ == "__main__":
    _init()