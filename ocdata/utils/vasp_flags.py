# OC20
# NOTE: this is the setting for slab and adslab
VASP_FLAGS = {
    "ibrion": 2,
    "nsw": 2000,
    "isif": 0,
    "isym": 0,
    "lreal": "Auto",
    "ediffg": -0.03,
    "symprec": 1e-10,
    "encut": 350.0,
    "laechg": True,
    "lwave": False,
    "ncore": 4,
    "gga": "RP",
    "pp": "PBE",
    "xc": "PBE",
}
# This is the setting for bulk optmization.
# Only use when expanding the bulk_db with other crystal structures.
BULK_VASP_FLAGS = {
    "ibrion": 1,
    "nsw": 100,
    "isif": 7,
    "isym": 0,
    "ediffg": 1e-08,
    "encut": 500.0,
    "kpts": (10, 10, 10),
    "prec": "Accurate",
    "gga": "RP",
    "pp": "PBE",
    "lwave": False,
    "lcharg": False,
}

SOLVENT_FLAGS = {
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