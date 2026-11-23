"""Wie soll mit dem ArUco-Marker umgegangen werden, sobald dieser erkannt wurde.

Zeichnet Objekte über die Marker und ermöglicht das Senden von Pakten wenn der 
Abstand zu einem Marker unter einen gewissen Wert fällt. 

Abhängigkeiten: 
    numpy, cv2, tcp_test, json

"""           
import numpy as np
import cv2
import tcp_test as tt
import json
import cv2

MIN_DEPTH = 0.03 # Objektpunkte näher als 3 cm an der Kamera weden nicht 
# gezeichnet
MAZE_W, MAZE_H = 10, 5

WALL_HEIGHT = 0.05
MARKER_HEIGHT = 0.025
OBJECT_HEIGHT = 0.003 # Höhe des Objetk-Nullpunkts über dem Boden (m) 

def estimate_marker_pose(corners, index, marker_3d_edges, camera_matrix,
                         dist_coeffs):
    """
    Args: 
        corners: Eckpunkte aller erkannten Marker im Bild (Ausgabe von
            detectMarkers). Pro Marker ein Array der Form (1, 4, 2) mit den
            vier Ecken als Pixelkoordinaten (x, y).
        index: Position des Zielmarkers in corners (und ids). Wählt aus,
            welcher der erkannten Marker verarbeitet wird.
        marker_3d_edges: Die vier Ecken des Markers in 3D, Array der 
            Form (4, 3),float32, in Metern. Liegen in der Ebene Z = 0 in der Reihenfolge
            oben links, oben rechts, unten rechts, unten links.
        camera_matrix: Kamera-Matrix (3, 3) mit Brennweiten und Bildmittelpunkt
            in Pixeln, aus der Kalibrierung.
        dist_coeffs: Verzerrungskoeffizienten (1, 5) in der Reihenfolge
            [k1, k2, p1, p2, k3], aus derselben Kalibrierung.

    Returns: 
        rvec: Wie der Marker zur Kamera gedreht ist
        tvec: Wie der Marker zur Kamera verschoben ist
        distance: Abstand zwischen Kamera und Marker in Metern (Tiefe in
            Blickrichtung der Kamera) als float. None, wenn die Lage des
            Markers nicht bestimmt werden konnte.
        corners_2d:Die vier Ecken eines einzelnen Markers im Kamerabild in 
            Pixeln 

    """
    
    corners_2d = corners[index][0].astype(np.float32)

    # Position des Markers relativ zur Kamera bestimmen
    success, rvec, tvec = cv2.solvePnP(
        marker_3d_edges, corners_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not success: 
        return None

    distance = float(tvec[2][0])

    return rvec, tvec, distance, corners_2d
    

def draw_object(frame, rvec, tvec, corners_2d, crystal_3d_pts, faces, edges,
                 camera_matrix, dist_coeffs):
    """
    Args: 
        crystal_3d_pts: Eckpunkte des 3D-Objekts, Array der Form (N, 3),float32.
            Koordinaten in Metern im Koordinatensystem des Markers.
        camera_matrix: Kamera-Matrix (3, 3) mit Brennweiten und Bildmittelpunkt
            in Pixeln, aus der Kalibrierung.
        dist_coeffs: Verzerrungskoeffizienten (1, 5) in der Reihenfolge
            [k1, k2, p1, p2, k3], aus derselben Kalibrierung.
        frame: Aktuelles Kamerabild als BGR-Array (Höhe, Breite, 3), uint8.
            Wird direkt verändert, das Objekt wird hineingezeichnet.
        faces: Dreiecke des 3D-Objekts als Liste von Listen, jede mit drei
            Indizes in crystal_3d_pts.
        edges: Kanten des 3D-Objekts, Array der Form (E, 2) mit je zwei
            Indizes in crystal_3d_pts.
        corners_2d: Die vier Ecken eines einzelnen Markers im Kamerabild in 
            Pixeln

    Returns: 
        
    """
    # Tiefe aller Objektpunkte im Kamerakoordinatendaten. Liegt einer zu nah an
    # der Kamera oder dahinter, wird nicht gezeichnet (sonst absurde Pixelwerte)
    rot, _ = cv2.Rodrigues(rvec)
    cam_pts = crystal_3d_pts @ rot.T + tvec.reshape(1,3) # rot.T dreht die Ecken aus Marker-Koordinatensystem in das Kamera-Koordinatensystem / + tvec verschiebt sie um den Weg von der Kamera zum Marker
    if cam_pts[:, 2].min() < MIN_DEPTH:
        return 

    img_pts, _ = cv2.projectPoints(
        crystal_3d_pts, rvec, tvec, camera_matrix, dist_coeffs
    )
    img_pts = img_pts.reshape(-1, 2)
    h, w = frame.shape[:2]
    if (not np.isfinite(img_pts).all() or
            np.abs(img_pts[:, 0] - w / 2).max() > 2 * w or
            np.abs(img_pts[:, 1] - h / 2).max() > 2 * h):
        return                                       # weit außerhalb, Verzerrung unzuverlässig
    img_pts = np.int32(img_pts) 

    overlay = frame.copy()
    for face in faces:
        pts = img_pts[face]
        cv2.fillPoly(overlay, [pts], (20, 20, 180))

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    for edge in edges:
        pt1 = tuple(img_pts[edge[0]])
        pt2 = tuple(img_pts[edge[1]])
        cv2.line(frame, pt1, pt2, (0, 0, 100), 1, cv2.LINE_AA)

    cv2.polylines(frame, [np.int32(corners_2d)], True, (0, 255, 0), 2)

def send_packet_by_distance(dist, triggered, value, send_below=0.5, reset_above=0.6):
    """Sendet ein Netzwerkpaket an den ESP32 abhängig vom Abstand zum Marker.
    
    Wenn sich der Roboter einem Marker nähert sollen verschiedene Aktionen 
    ausgelöst werden, der Roboter sammelt Powerups quasi ein. 

    Args: 
        dist: gemessener Abstand in Metern oder None (kein Marker).
        triggered: ob für diese Annäherung schon gesendet wurde.

    Returns: 
        triggered: Gibt den neuen Zustand von triggered zurück.
    """
   
    if dist is not None and dist < send_below and not triggered:
        try:
            tt.send_packet(value)
        except Exception as e:
            print(f"ESP kann nicht erreicht werden{e}")
        return True
    if dist is None or dist > reset_above:
        return False
    return triggered

def load_labyrinth_config(path: str) -> dict:
    """Einmal beim Start aufrufen. Wirft ValueError bei ungültiger Konfiguration."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    factor = data.get("factor")
    if not isinstance(factor, (int, float)) or factor <= 0:
        raise ValueError(f"factor muss > 0 sein, ist {factor!r}")
    for key, value in data.get("markers", {}).items():
        if len(value) != 3 or value[2] not in ("n", "e", "s", "w"):
            raise ValueError(f"Marker {key}: ungültiger Eintrag {value!r}")
    return data

def calc_pac_pos(rvec, tvec, marker_id, config: dict):
    """Zelle (x, y) der Kamera aus der Pose eines Wandmarkers.

    Der Buchstabe im Marker-Eintrag gibt an, in welche Himmelsrichtung die
    Vorderseite des Markers zeigt (x = Osten, y = Norden).
    """
    key = str(int(marker_id))
    if key not in config["markers"]:
        raise ValueError(key)
    mx, my, direction = config["markers"][key]
    factor = config["factor"]

    rot, _ = cv2.Rodrigues(rvec)          # Wandelt rvec (Rotationsvektor) in eine 3x3 Rotationsmatrix um -> Ermöglicht direktes verrechnen mit Punkten durch das Skalarprodukt
    cam = (-rot.T @ tvec).ravel()        # setzt die Punkte aus dem Kamerakoordinatensystem in das Markerkoordinatensystem (x rechts, y oben, z aus der Wand) -> rvec und tvec gehen ausgehend vom Kamerakoordinatensystem

    match direction:
        case "n": 
            offset = (cam[0],  -cam[2])
        case "e": 
            offset = ( -cam[2],  -cam[0])
        case "s": 
            offset = ( -cam[0], cam[2])
        case "w": 
            offset = (cam[2], cam[0])

    x = round((mx * factor + offset[0]) / factor)
    y = round((my * factor + offset[1]) / factor)
    if not (0 <= x < MAZE_W and 0 <= y < MAZE_H):
        raise ValueError(f"Zelle ({x}, {y}) außerhalb des Labyrinths")
    return x, y

WALL_AXES = {   # Spalten: Marker-x, Marker-y, Marker-z in Labyrinth-Koordinaten
    "n": [[ 1, 0,  0], [0, 0, 1], [ 0, -1, 0]],
    "e": [[ 0, -1, 0], [0, 0, 1], [-1,  0, 0]],
    "s": [[-1, 0,  0], [0, 0, 1], [ 0,  1, 0]],
    "w": [[ 0, 1,  0], [0, 0, 1], [ 1,  0, 0]],
}

def marker_pose_in_maze(marker_id, config):
    """Marker Position umrechnen in Globale (Labyrinth) Position"""

    marker_x, marker_y, wall = config["markers"][str(marker_id)]
    faktor = config["factor"]
    R_wm = np.array(WALL_AXES[wall], dtype=np.float64).T # Gibt die Drehmatrix R_wm an (entsprechend WALL_AXES) wie stehen die Achsen des Markerkoordinatensystems im Weltkoordinatensystem
    p_wm = np.array([marker_x * faktor, marker_y * faktor, MARKER_HEIGHT]) # Gibt die Verschiebung an da die Koordinaten in Weltkoordinaten vorliegen und wir es genau brauchen, rechnen wir sie in m (durch den Faktor)
    return R_wm, p_wm 

def maze_pose_in_camera(rvec, tvec, R_wm, p_wm):
    """Aus Drehung und Verschiebung der Kamera zum Marker sowie vom Marker zum 
    globalen Labyrinth lässt sich die Drehung und Verschiebung der Kamera zum 
    globalen Labyrinth besitmmen"""

    R_km, _ = cv2.Rodrigues(rvec) # Rodrigues wandelt einfach einen Drehvektor in eine Drehmatrix und umgekehrt (Drehvektor besser zum speichern (nur 3 Zahlen) - Drehmatrix besser zum Rechnen)
    R_kw = R_km @ R_wm.T # Durch Multiplikation (hier durch @ Matrixweit) beider Drehmatrizen K->M und M->W lässt sich die Drehmatrize K->W bilden
    tvec_w = tvec.reshape(3) - R_kw @ p_wm
    rvec_w, _ = cv2.Rodrigues(R_kw)

    return rvec_w, tvec_w

def object_pose_in_camera(rvec_w, tvec_w, cell_x, cell_y, config, height=OBJECT_HEIGHT):
    """Pose eines Objekts in Zelle (cell_x, cell_y) aus Sicht der Kamera."""
    R_kw, _ = cv2.Rodrigues(rvec_w)
    f = config["factor"]
    ziel = np.array([cell_x * f, cell_y * f, height]) # Ziel festlegen
    tvec_obj = tvec_w + R_kw @ ziel # Objekt auf Zielzelle verschieben
    return rvec_w, tvec_obj 

def draw_cell_grid(frame, rvec_w, tvec_w, config, camera_matrix, dist_coeffs):
    """Debug: alle Zellmitten am Boden als Punkt mit Koordinaten einzeichnen."""
    f = config["factor"]
    R_kw, _ = cv2.Rodrigues(rvec_w)
    for x in range(MAZE_W):
        for y in range(MAZE_H):
            p = np.array([x * f, y * f, 0.0])
            if (R_kw @ p + tvec_w)[2] < MIN_DEPTH:      # hinter der Kamera
                continue
            px, _ = cv2.projectPoints(p.reshape(1, 3), rvec_w, tvec_w,
                                      camera_matrix, dist_coeffs)
            u, v = px.ravel()
            h, w = frame.shape[:2]
            if not (np.isfinite(u) and np.isfinite(v) and 0 <= u < w and 0 <= v < h):
                continue                                 # außerhalb des Bildes
            u, v = int(u), int(v) 
            cv2.circle(frame, (u, v), 4, (0, 255, 255), -1)
            cv2.putText(frame, f"{x},{y}", (u + 5, v - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

def camera_in_maze(rvec_w, tvec_w): # Position wird anhand der Formel P_K = R_kw * P_W + tvec_w ausgerechent wobei P_K 0 ist da die Position der Kamera im Kamerakoordinatensystem selbst 0 ist
    """Position der Kamera in Labyrinth-Koordinaten (Meter, x Osten, y Norden, z oben)."""
    R_kw, _ = cv2.Rodrigues(rvec_w)
    return -R_kw.T @ tvec_w.reshape(3)

def cross(o, p, q):
    """Auf welcher Seite der Linie o→p liegt q? >0 links, <0 rechts, 0 auf der Linie."""
    return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

def cut_position(a, b, c, d):
    # 1. Liegen die Wandenden c, d auf verschiedenen Seiten der Sichtlinie a→b?  
    cut_c = cross(a, b, c)
    cut_d = cross(a, b, d)
    # 2. Liegen Kamera a und Objekt b auf verschiedenen Seiten der Wand c→d?   
    cut_a = cross(c, d, a)
    cut_b = cross(c, d, b)

    if cut_c * cut_d > 0 or cut_a * cut_b > 0:   # gleiches Vorzeichen → kein Schnitt
        return None
    if cut_a == cut_b:                           # parallel / auf einer Linie
        return None
    return cut_a / (cut_a - cut_b)               # s: 0 = Kamera, 1 = Objekt 

def is_visible(cam, obj, walls):
    """cam, obj: 3D-Punkte im Labyrinth. walls: Liste von Strecken ((x1,y1),(x2,y2))."""
    for c, d in walls:
        s = cut_position(cam[:2], obj[:2], c, d)
        if s is None:
            continue
        h = cam[2] + s * (obj[2] - cam[2])   # Höhe der Sichtlinie an der Wand
        if h <= WALL_HEIGHT:
            return False                     # Wand verdeckt
    return True

def wall_segments(config):
    # Aus Wänden Strecken machen
    f = config["factor"]
    segs = []
    for x, y, o in config["walls"].values():
        if o == "v": # Nord-Süd: eine Zelle lang in y-Richtung
            segs.append(((x * f, (y - 0.5) * f), (x * f, (y + 0.5) * f)))
        else: # h: Ost-West
            segs.append((((x - 0.5) * f, y * f), ((x + 0.5) * f, y * f)))
    return segs

class PoseSmoother:
    """Exponentielle Glättung einer Pose. alpha: 0 = eingefroren, 1 = keine Glättung."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.R = None
        self.t = None

    def update(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        t = np.ravel(tvec).astype(float)
        if self.R is None:
            self.R, self.t = R, t
        else:
            a = self.alpha
            self.t = (1 - a) * self.t + a * t
            M = (1 - a) * self.R + a * R
            U, _, Vt = np.linalg.svd(M)
            self.R = U @ Vt
        r, _ = cv2.Rodrigues(self.R)
        return r, self.t.copy()

    def reset(self):
        self.R = self.t = None