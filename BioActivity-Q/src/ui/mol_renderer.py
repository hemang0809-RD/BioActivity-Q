"""2D molecule rendering as SVG/PNG for web display."""
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import io
import base64


def smiles_to_svg(smiles, width=350, height=250, highlight_bits=None):
    """
    Render a molecule as SVG string for embedding in HTML.
    highlight_bits: optional list of Morgan bit indices to highlight (e.g. from SHAP).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "<p style='color:red'>Invalid SMILES</p>"

    AllChem.Compute2DCoords(mol)

    highlight_atoms = []
    if highlight_bits:
        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bit_info)
        for bit in highlight_bits:
            if bit in bit_info:
                for atom_idx, radius in bit_info[bit]:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                    amap = {}
                    Chem.PathToSubmol(mol, env, atomMap=amap)
                    highlight_atoms.extend(amap.keys())

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().addAtomIndices = False

    if highlight_atoms:
        colors = {a: (1.0, 0.8, 0.8) for a in highlight_atoms}
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(set(highlight_atoms)),
            highlightAtomColors=colors,
        )
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def smiles_to_base64_png(smiles, size=(350, 250)):
    """Returns base64-encoded PNG for <img src='data:image/png;base64,...'>."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
