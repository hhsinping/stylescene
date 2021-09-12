from pathlib import Path
import socket
import platform
import getpass

Train = False
Test_style = [3,4]

tat_root = Path("../../colmap_tat/")
styleroot = "../../style_data/"
HOSTNAME = socket.gethostname()
PLATFORM = platform.system()
USER = getpass.getuser()
train_device = "cuda:0"
eval_device = "cuda:0"
dtu_root = None
colmap_bin_path = None
lpips_root = None

tat_train_sets = [
    "training/Barn",
    "training/Caterpillar",
    "training/Church",
    "training/Ignatius",
    "training/Meetingroom",
    "intermediate/Family",
    "intermediate/Francis",
    "intermediate/Horse",
    "intermediate/Lighthouse",
    "intermediate/Panther",
    "advanced/Auditorium",
    "advanced/Ballroom",
    "advanced/Museum",
    "advanced/Temple",
    "advanced/Courtroom",
    "advanced/Palace",
]

tat_eval_sets = [
    "training/Truck",
    "intermediate/M60",
    "intermediate/Playground",
    "intermediate/Train",
]

tat_eval_tracks = {}
tat_eval_tracks['training/Truck'] = [172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]
tat_eval_tracks['intermediate/M60'] = [94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
tat_eval_tracks['intermediate/Playground'] = [221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252]
tat_eval_tracks['intermediate/Train'] = [174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248]
