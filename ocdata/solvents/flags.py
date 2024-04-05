BASE_FLAGS = {
    "prec": "Normal",
    "gga": "RP",
    "pp": "PBE",
    "xc": "PBE",
    "ivdw": 11,
    "encut": 400.0,
    "ediff": 1e-6,
    "nelm": 100,
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
    "dipol": [0.5, 0.5, 0.5],
}

OPT_FLAGS = {"ibrion": 2, "nsw": 0}

MD_FLAGS = {
    "ibrion": 0,
    "nsw": 500,
    "smass": 0,
    "tebeg": 300,
    "potim": 1,
}

ML_FLAGS = {"ML_LMLFF": True, "ML_MODE": "train", "ML_EPS_LOW": 1e-7}
