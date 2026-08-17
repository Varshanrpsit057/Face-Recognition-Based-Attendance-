import csv
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from config import cfg
from src.logger import get_logger
from src.utils import get_timestamp

logger = get_logger(__name__)

@dataclass
class AttendanceRecord:
    student_id: str
    name: str
    timestamp: str
    confidence: float
    track_id: int
    camera_id: str
    recognition_latency: float

class TemporalVoter:
    def __init__(self, m: int, n: int, min_confidence: float):
        self.m = m
        self.n = n
        self.min_confidence = min_confidence
        self._vote_buffer: Dict[int, List[Tuple[str, float]]] = {}

    def vote(self, track_id: int, student_id: str, confidence: float) -> Optional[Tuple[str, float]]:
        if track_id not in self._vote_buffer:
            self._vote_buffer[track_id] = []
            
        self._vote_buffer[track_id].append((student_id, confidence))
        
        if len(self._vote_buffer[track_id]) > self.n:
            self._vote_buffer[track_id].pop(0)
            
        if len(self._vote_buffer[track_id]) == self.n:
            votes = self._vote_buffer[track_id]
            id_counts = {}
            id_confs = {}
            for sid, conf in votes:
                if conf >= self.min_confidence:
                    id_counts[sid] = id_counts.get(sid, 0) + 1
                    id_confs.setdefault(sid, []).append(conf)
            
            for sid, count in id_counts.items():
                if count >= self.m:
                    avg_conf = sum(id_confs[sid]) / len(id_confs[sid])
                    return (sid, avg_conf)
                    
        return None

    def reset_track(self, track_id: int) -> None:
        if track_id in self._vote_buffer:
            del self._vote_buffer[track_id]

    def clear(self) -> None:
        self._vote_buffer.clear()

class AttendanceEngine:
    def __init__(self, config=None):
        self.config = config if config else cfg.attendance
        self._records: List[AttendanceRecord] = []
        self._cooldown_tracker: Dict[str, float] = {}
        self._voter = TemporalVoter(cfg.voting.m, cfg.voting.n, cfg.voting.min_confidence)
        
        self.attendance_dir = Path(cfg.paths.attendance_dir)
        self.attendance_dir.mkdir(parents=True, exist_ok=True)

    def mark_attendance(self, student_id: str, confidence: float, track_id: int, latency: float) -> Optional[AttendanceRecord]:
        now = datetime.now().timestamp()
        
        if student_id in self._cooldown_tracker:
            if (now - self._cooldown_tracker[student_id]) < self.config.cooldown_seconds:
                return None
                
        record = AttendanceRecord(
            student_id=student_id,
            name=student_id, 
            timestamp=get_timestamp(),
            confidence=confidence,
            track_id=track_id,
            camera_id=self.config.camera_id,
            recognition_latency=latency
        )
        self._records.append(record)
        self._cooldown_tracker[student_id] = now
        logger.info(f"Attendance marked for {student_id} (conf: {confidence:.2f})")
        return record

    def process_recognition(self, track_id: int, student_id: Optional[str], confidence: float, latency: float) -> Optional[AttendanceRecord]:
        if student_id is None:
            return None
            
        result = self._voter.vote(track_id, student_id, confidence)
        if result:
            confirmed_id, avg_conf = result
            return self.mark_attendance(confirmed_id, avg_conf, track_id, latency)
        return None

    def get_records(self) -> List[AttendanceRecord]:
        return self._records

    def get_today_records(self) -> List[AttendanceRecord]:
        today = datetime.now().strftime("%Y%m%d")
        return [r for r in self._records if r.timestamp.startswith(today)]

    def get_record_count(self) -> int:
        return len(self._records)

    def get_today_count(self) -> int:
        return len(self.get_today_records())

    def export_csv(self, path: Path = None) -> Path:
        if path is None:
            path = self.attendance_dir / f"attendance_{get_timestamp()}.csv"
            
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Student ID', 'Name', 'Timestamp', 'Confidence', 'Track ID', 'Camera ID', 'Latency'])
            for r in self._records:
                writer.writerow([r.student_id, r.name, r.timestamp, r.confidence, r.track_id, r.camera_id, r.recognition_latency])
        return path

    def export_excel(self, path: Path = None) -> Path:
        import xlsxwriter
        if path is None:
            path = self.attendance_dir / f"attendance_{get_timestamp()}.xlsx"
            
        workbook = xlsxwriter.Workbook(str(path))
        worksheet = workbook.add_worksheet()
        
        headers = ['Student ID', 'Name', 'Timestamp', 'Confidence', 'Track ID', 'Camera ID', 'Latency']
        for col, h in enumerate(headers):
            worksheet.write(0, col, h)
            
        for row, r in enumerate(self._records, start=1):
            worksheet.write(row, 0, r.student_id)
            worksheet.write(row, 1, r.name)
            worksheet.write(row, 2, r.timestamp)
            worksheet.write(row, 3, r.confidence)
            worksheet.write(row, 4, r.track_id)
            worksheet.write(row, 5, r.camera_id)
            worksheet.write(row, 6, r.recognition_latency)
            
        workbook.close()
        return path

    def export_json(self, path: Path = None) -> Path:
        if path is None:
            path = self.attendance_dir / f"attendance_{get_timestamp()}.json"
            
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([r.__dict__ for r in self._records], f, indent=4)
        return path

    def export_sqlite(self, path: Path = None) -> Path:
        if path is None:
            path = self.attendance_dir / f"attendance_{get_timestamp()}.db"
            
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                name TEXT,
                timestamp TEXT,
                confidence REAL,
                track_id INTEGER,
                camera_id TEXT,
                latency REAL
            )
        ''')
        
        for r in self._records:
            cursor.execute('''
                INSERT INTO attendance (student_id, name, timestamp, confidence, track_id, camera_id, latency)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (r.student_id, r.name, r.timestamp, r.confidence, r.track_id, r.camera_id, r.recognition_latency))
            
        conn.commit()
        conn.close()
        return path

    def export_all(self) -> Dict[str, Path]:
        return {
            'csv': self.export_csv(),
            'excel': self.export_excel(),
            'json': self.export_json(),
            'sqlite': self.export_sqlite()
        }

    def clear(self) -> None:
        self._records.clear()
        self._cooldown_tracker.clear()
        self._voter.clear()

    def get_attendance_rate(self) -> float:
        return 0.0

    def get_student_attendance(self, student_id: str) -> List[AttendanceRecord]:
        return [r for r in self._records if r.student_id == student_id]
