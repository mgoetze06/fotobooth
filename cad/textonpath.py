
from math import degrees, atan2
import cadquery as cq


def _faceOnWire(self, path: cq.Wire) -> cq.Face:
    """Reposition a face from alignment to the x-axis to the provided path"""
    path_length = path.Length()

    bbox = self.BoundingBox()
    face_bottom_center = cq.Vector((bbox.xmin + bbox.xmax) / 2, 0, 0)
    relative_position_on_wire = face_bottom_center.x / path_length
    wire_tangent = path.tangentAt(relative_position_on_wire)
    wire_angle = degrees(atan2(wire_tangent.y, wire_tangent.x))
    wire_position = path.positionAt(relative_position_on_wire)

    return self.rotate(
        face_bottom_center, face_bottom_center + cq.Vector(0, 0, 1), wire_angle
    ).translate(wire_position - face_bottom_center)


cq.Face.faceOnWire = _faceOnWire


def textOnWire(txt: str, fontsize: float, distance: float, path: cq.Wire) -> cq.Solid:
    """Create 3D text with a baseline following the given path"""
    linear_faces = (
        cq.Workplane("XY")
        .text(
            txt=txt,
            fontsize=fontsize,
            distance=distance,
            halign="left",
            valign="bottom",
        )
        .faces("<Z")
        .vals()
    )
    faces_on_path = [f.faceOnWire(path) for f in linear_faces]
    return cq.Compound.makeCompound(
        [cq.Solid.extrudeLinear(f, cq.Vector(0, 0, 1)) for f in faces_on_path]
    )


path = cq.Edge.makeThreePointArc(
    cq.Vector(0, 0, 0), cq.Vector(30, 10, 0), cq.Vector(60, 0, 0)
)
text_on_path = textOnWire(
    txt="The quick brown fox jumped over the lazy dog",
    fontsize=5,
    distance=1,
    path=path,
)
if "show_object" in locals():
    show_object(text_on_path, name="text_on_path")
    show_object(path, name="path")