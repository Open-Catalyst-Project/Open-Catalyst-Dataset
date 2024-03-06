BASE_FLAGS = {
    "prec": "Normal",
    "gga": "RP",
    "pp": "PBE",
    "xc": "PBE",
    "ivdw": 11,
    "encut": 400.0,
    "ediff": 1e-4,
    "nelm": 200,
    "ismear": 0,
    "sigma": 0.1,
    "lwave": False,
    "lcharg": False,
    "isif": 0,
    "ispin": 2,
    "algo": "Fast",
    "idipol": 3,
    "ldipol": True,
    "lasph": True,
    "lreal": "Auto",
    "ncore": 4,
}

OPT_FLAGS = {"ibrion": 2, "nsw": 0}

MD_FLAGS = {
    "ibrion": 0,
    "nsw": 500,
    "smass": 0,
    "tebeg": 300,
    "potim": 1,
}
