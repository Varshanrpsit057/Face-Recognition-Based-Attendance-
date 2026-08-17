import numpy as np
from typing import List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from src.detector import Detection
from config import TrackerBackend

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    age: int
    lost_frames: int
    velocity: Tuple[float, float]
    is_confirmed: bool

class TrackerInterface(ABC):
    @abstractmethod
    def update(self, detections: List[Detection], image_shape: Tuple[int, int]) -> List[Track]: ...
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def name(self) -> str: ...

def iou_batch(bboxes1, bboxes2):
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))
    b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]
    
    inter_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2)
    
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union_area = b1_area[:, None] + b2_area - inter_area
    return inter_area / np.maximum(union_area, 1e-6)

class STrack:
    _count = 0
    def __init__(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        STrack._count += 1
        self.track_id = STrack._count
        self.age = 0
        self.lost_frames = 0
        self.is_confirmed = False
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self._convert_bbox_to_z(bbox).reshape(4, 1)

    def _convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2
        y = bbox[1] + h / 2
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r])
    
    def _convert_x_to_bbox(self, x):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return [int(x[0] - w/2), int(x[1] - h/2), int(x[0] + w/2), int(x[1] + h/2)]

    def predict(self):
        if self.kf.x[7-1, 0] + self.kf.x[3-1, 0] <= 0:
            self.kf.x[7-1, 0] *= 0.0
        self.kf.predict()

    def update(self, bbox, confidence):
        self.age += 1
        self.lost_frames = 0
        self.confidence = confidence
        self.kf.update(self._convert_bbox_to_z(bbox))
        self.bbox = self._convert_x_to_bbox(self.kf.x)
        if self.age >= 3:
            self.is_confirmed = True

    def mark_lost(self):
        self.lost_frames += 1

class ByteTracker(TrackerInterface):
    def __init__(self):
        self.tracks = []
        self.high_thresh = 0.5
        self.low_thresh = 0.1
        self.match_thresh = 0.8
        self.max_lost = 30

    def name(self) -> str: return "ByteTrack"
    
    def reset(self) -> None:
        self.tracks = []
        STrack._count = 0

    def _associate(self, detections, tracks, threshold):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        track_bboxes = np.array([t.bbox for t in tracks])
        det_bboxes = np.array([d.bbox for d in detections])
        
        ious = iou_batch(track_bboxes, det_bboxes)
        cost_matrix = 1 - ious
        
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        
        matches, unmatched_dets, unmatched_tracks = [], [], []
        for r, c in zip(row_inds, col_inds):
            if cost_matrix[r, c] > threshold:
                unmatched_tracks.append(r)
                unmatched_dets.append(c)
            else:
                matches.append((r, c))
                
        unmatched_dets.extend(list(set(range(len(detections))) - set(col_inds)))
        unmatched_tracks.extend(list(set(range(len(tracks))) - set(row_inds)))
        
        return matches, unmatched_dets, unmatched_tracks

    def update(self, detections: List[Detection], image_shape: Tuple[int, int]) -> List[Track]:
        for t in self.tracks: t.predict()
        
        dets_high = [d for d in detections if d.confidence >= self.high_thresh]
        dets_low = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]
        
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed]
        unconfirmed_tracks = [t for t in self.tracks if not t.is_confirmed]
        
        matches_high, u_dets_high, u_tracks = self._associate(dets_high, confirmed_tracks, self.match_thresh)
        for r, c in matches_high:
            confirmed_tracks[r].update(dets_high[c].bbox, dets_high[c].confidence)
            
        remaining_tracks = [confirmed_tracks[i] for i in u_tracks]
        matches_low, u_dets_low, u_tracks_low = self._associate(dets_low, remaining_tracks, 0.5)
        for r, c in matches_low:
            remaining_tracks[r].update(dets_low[c].bbox, dets_low[c].confidence)
            
        for i in u_tracks_low:
            remaining_tracks[i].mark_lost()
            
        matches_unc, u_dets_unc, u_tracks_unc = self._associate([dets_high[i] for i in u_dets_high], unconfirmed_tracks, 0.7)
        for r, c in matches_unc:
            unconfirmed_tracks[r].update(dets_high[u_dets_high[c]].bbox, dets_high[u_dets_high[c]].confidence)
            
        for i in u_tracks_unc:
            unconfirmed_tracks[i].mark_lost()
            
        new_tracks = []
        for i in u_dets_high:
            if i not in [c for r, c in matches_unc]:
                new_tracks.append(STrack(dets_high[i].bbox, dets_high[i].confidence))
                
        self.tracks = [t for t in confirmed_tracks + unconfirmed_tracks + new_tracks if t.lost_frames < self.max_lost]
        
        res = []
        for t in self.tracks:
            if t.is_confirmed:
                res.append(Track(t.track_id, tuple(t.bbox), t.confidence, t.age, t.lost_frames, (0.0, 0.0), t.is_confirmed))
        return res

class SORTTracker(ByteTracker):
    def name(self) -> str: return "SORT"
    def update(self, detections: List[Detection], image_shape: Tuple[int, int]) -> List[Track]:
        for t in self.tracks: t.predict()
        matches, u_dets, u_tracks = self._associate(detections, self.tracks, self.match_thresh)
        for r, c in matches:
            self.tracks[r].update(detections[c].bbox, detections[c].confidence)
        for i in u_tracks:
            self.tracks[i].mark_lost()
        for i in u_dets:
            self.tracks.append(STrack(detections[i].bbox, detections[i].confidence))
        self.tracks = [t for t in self.tracks if t.lost_frames < self.max_lost]
        res = []
        for t in self.tracks:
            if t.is_confirmed:
                res.append(Track(t.track_id, tuple(t.bbox), t.confidence, t.age, t.lost_frames, (0.0, 0.0), t.is_confirmed))
        return res

class TrackerFactory:
    @staticmethod
    def create(backend: TrackerBackend = None) -> TrackerInterface:
        if backend == TrackerBackend.SORT: return SORTTracker()
        return ByteTracker()
