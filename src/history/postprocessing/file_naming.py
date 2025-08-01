import os

class FileNaming:
    def __init__(self, file: str):
        filename = os.path.basename(file)

        self.splited_filename = filename.split("_")

        assert len(self.splited_filename) > 8

        self.author = self.splited_filename[0]
        self.site = self.splited_filename[1]
        self.dataset = self.splited_filename[2]
        self.images = self.splited_filename[3]
        self.camera_used = self.splited_filename[4]
        self.gcp_used = self.splited_filename[5]
        self.pointcloud_coregistration = self.splited_filename[6]
        self.mtp_adjustment = self.splited_filename[7]


        assert self.site in ["CG", "IL"]
        assert self.dataset in ["AI", "MC", "PC"]
        assert self.images in ["RA", "PP"]
        assert self.camera_used in ["CY", "CN"]
        assert self.gcp_used in ["GY", "GN"]
        assert self.pointcloud_coregistration in ["PY", "PN"]
        assert self.mtp_adjustment in ["MY", "MN"]



