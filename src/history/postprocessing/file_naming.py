import os


class FileNaming(dict):
    # Mappings for validation and optional human-readable names
    VALID_SITE = {"CG": "casa_grande", "IL": "iceland"}
    VALID_DATASET = {"AI": "aerial", "MC": "kh9mc", "PC": "kh9pc"}
    VALID_IMAGES = {"RA": "raw", "PP": "preprocessed"}
    VALID_CAMERA = {"CY": True, "CN": False}
    VALID_GCP = {"GY": True, "GN": False}
    VALID_COREG = {"PY": True, "PN": False}
    VALID_MTP = {"MY": True, "MN": False}
    VALID_PCTYPE = {"dense": "dense", "sparse": "sparse"}

    # Define the keys and their validation mapping in order
    KEYS_MAPPING = [
        ("author", None),
        ("site", VALID_SITE),
        ("dataset", VALID_DATASET),
        ("images", VALID_IMAGES),
        ("camera_used", VALID_CAMERA),
        ("gcp_used", VALID_GCP),
        ("pointcloud_coregistration", VALID_COREG),
        ("mtp_adjustment", VALID_MTP),
        ("pointcloud_type", VALID_PCTYPE),
    ]

    def __init__(self, file: str):
        super().__init__()  # initialize dict

        filename = os.path.basename(file)
        parts = filename.split("_")

        assert len(parts) > 8, f"File {file} has unexpected format"

        self["code"] = parts[0]

        # First part is always author
        self["author"] = parts[0]

        # For other codes, search in the filename
        remaining_parts = parts[1:]
        for key, mapping in self.KEYS_MAPPING[1:]:  # skip author
            if mapping is not None:
                # Find first part in remaining_parts that matches mapping
                found = next((p for p in remaining_parts if p in mapping), None)
                self[key] = mapping.get(found, "unknown")
                if found is None:
                    self["code"] += "_XX"
                else:
                    self["code"] += "_" + found

            else:
                self[key] = "unknown"
                self["code"] += "_XX"
