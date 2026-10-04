# PyInstaller spec: pyinstaller packaging/phoenix_stl.spec
# Produces dist/PhoenixSTLStudio/PhoenixSTLStudio.exe (one-folder build, starts fast).
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = os.path.abspath(os.path.join(SPECPATH, ".."))
src = os.path.join(root, "src")

datas = [
    (os.path.join(src, "phoenix_stl", "i18n"), os.path.join("phoenix_stl", "i18n")),
    (os.path.join(src, "phoenix_stl", "assets"), os.path.join("phoenix_stl", "assets")),
]
datas += collect_data_files("pyvista")
binaries = []
# Only the VTK modules the app really loads (recorded at runtime); bundling all
# of vtkmodules would add ~500 MB.
with open(os.path.join(SPECPATH, "vtk_modules.txt")) as f:
    vtk_modules = [line.strip() for line in f if line.strip()]
hiddenimports = (collect_submodules("phoenix_stl") + vtk_modules
                 + ["vtkmodules.util.numpy_support", "vtkmodules.qt.QVTKRenderWindowInteractor",
                    "pyvistaqt", "scipy.sparse.csgraph._validation", "manifold3d", "mapbox_earcut"])

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[src],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "IPython", "jupyter", "trame", "PyQt5", "PyQt6",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick",
              "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts",
              "PySide6.QtDataVisualization", "PySide6.QtPdf", "PySide6.QtSql", "PySide6.QtTest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PhoenixSTLStudio",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PhoenixSTLStudio", upx=False)
