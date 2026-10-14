import sqlite3
from pathlib import Path

import numpy as np


class FaceGallery:
    def __init__(self, db_path):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS faces ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT NOT NULL, "
            "embedding BLOB NOT NULL)"
        )
        self.conn.commit()

    def add(self, name, embedding):
        emb = np.asarray(embedding, dtype=np.float32)
        self.conn.execute(
            "INSERT INTO faces (name, embedding) VALUES (?, ?)",
            (name, emb.tobytes()),
        )
        self.conn.commit()

    def names(self):
        rows = self.conn.execute("SELECT DISTINCT name FROM faces").fetchall()
        return [r[0] for r in rows]

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def load(self):
        rows = self.conn.execute("SELECT name, embedding FROM faces").fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        names = [r[0] for r in rows]
        vectors = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        return names, vectors

    def match(self, embedding, threshold=0.35):
        names, vectors = self.load()
        if len(names) == 0:
            return "Unknown", 0.0
        emb = np.asarray(embedding, dtype=np.float32)
        sims = vectors @ emb
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < threshold:
            return "Unknown", score
        return names[best], score

    def close(self):
        self.conn.close()
