# run_download_voc.py  — FIXED VERSION
from ultralytics.utils.downloads import download
from pathlib import Path

out = Path("data/pascal_voc_raw")
out.mkdir(parents=True, exist_ok=True)

print("Downloading VOC 2007 from Ultralytics CDN...")
# download() auto-extracts — do NOT manually unzip after this
download(
    "https://ultralytics.com/assets/VOC2007.zip",
    dir=out
)

# Verify what was extracted
voc = out / "VOCdevkit" / "VOC2007"
if voc.exists():
    imgs = len(list((voc / "JPEGImages").glob("*.jpg")))
    anns = len(list((voc / "Annotations").glob("*.xml")))
    print(f"\n✅ Done! Images={imgs}  Annotations={anns}")
    print("→ Now run: python prepare_pascal.py")
else:
    # Ultralytics may extract to a different path — search for it
    found = list(out.rglob("JPEGImages"))
    if found:
        print(f"\n✅ Extracted to: {found[0].parent}")
        print("   Update VOC_ROOT in prepare_pascal.py to:", found[0].parent.parent)
    else:
        print("\n❌ Extraction failed — switching to torchvision fallback...")
        import torchvision.datasets as datasets
        print("Downloading trainval via torchvision (~460MB)...")
        datasets.VOCDetection(str(out), year='2007', image_set='trainval', download=True)
        print("Downloading test via torchvision (~430MB)...")
        datasets.VOCDetection(str(out), year='2007', image_set='test', download=True)
        imgs = len(list((voc / "JPEGImages").glob("*.jpg")))
        print(f"\n✅ Done! Images={imgs}")
        print("→ Now run: python prepare_pascal.py")
