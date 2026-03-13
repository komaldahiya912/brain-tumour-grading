# Copyright (c) 2026 Komal Dahiya
# GitHub: https://github.com/komaldahiya912/brain-tumour-grading
# Licensed under CC BY-NC 4.0 — Non-commercial use only.
# Credit must be given if this code is used or adapted.

"""
database.py — Brain Tumour Grading App
Fixed: WAL mode + threading.Lock + check_same_thread=False
       + conn.close() in finally + both pipeline tables
Extended: extra patient fields + MRI image stored as BLOB
"""

import sqlite3
import json
import threading
import os
from datetime import datetime
import numpy as np

_db_lock = threading.Lock()


class PredictionDatabase:
    def __init__(self, db_path="/tmp/predictions.db"):
        self.db_path = db_path
        self.init_db()

    def _connect(self):
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init_db(self):
        conn = None
        try:
            conn = self._connect()
            cur = conn.cursor()

            # Pipeline 1 — MRI predictions
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name     TEXT    NOT NULL,
                    patient_id       TEXT,
                    phone            TEXT,
                    gender           TEXT,
                    dob              TEXT,
                    upload_date      TEXT    NOT NULL,
                    image_path       TEXT    NOT NULL,
                    image_blob       BLOB,
                    tumor_present    INTEGER NOT NULL,
                    predicted_grade  INTEGER NOT NULL,
                    grade_confidence REAL    NOT NULL,
                    tumor_area       REAL    NOT NULL,
                    results_json     TEXT    NOT NULL
                )
            """)

            # Pipeline 2 — Clinical predictions
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clinical_predictions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name    TEXT    NOT NULL,
                    patient_id      TEXT,
                    phone           TEXT,
                    gender          TEXT,
                    dob             TEXT,
                    upload_date     TEXT    NOT NULL,
                    idh1            INTEGER NOT NULL,
                    age             REAL    NOT NULL,
                    pten            INTEGER NOT NULL,
                    egfr            INTEGER NOT NULL,
                    atrx            INTEGER NOT NULL,
                    predicted_class TEXT    NOT NULL,
                    confidence      REAL    NOT NULL,
                    raw_output      REAL    NOT NULL,
                    lgg_probability REAL    NOT NULL,
                    gbm_probability REAL    NOT NULL
                )
            """)

            conn.commit()
        except Exception as e:
            print(f"[DB] init_db error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    # ════════════════════════════════════════════════════════════════
    # PIPELINE 1 — MRI
    # ════════════════════════════════════════════════════════════════
    def save_prediction(self, patient_info, image_path, results, image_bytes=None):
        """
        patient_info : dict {name, patient_id, phone, gender, dob}
        image_bytes  : raw bytes of uploaded image (stored as BLOB)
        """
        conn = None
        with _db_lock:
            try:
                mask_name = os.path.splitext(os.path.basename(image_path))[0] + "_mask.npy"
                mask_dir  = os.path.join("static", "masks")
                os.makedirs(mask_dir, exist_ok=True)
                mask_path = os.path.join(mask_dir, mask_name)
                np.save(mask_path, results["tumor_mask"])

                serializable = results.copy()
                serializable["tumor_mask"] = mask_path

                conn = self._connect()
                cur  = conn.cursor()
                cur.execute(
                    """INSERT INTO predictions
                       (patient_name, patient_id, phone, gender, dob,
                        upload_date, image_path, image_blob,
                        tumor_present, predicted_grade, grade_confidence,
                        tumor_area, results_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        patient_info.get("name", ""),
                        patient_info.get("patient_id", ""),
                        patient_info.get("phone", ""),
                        patient_info.get("gender", ""),
                        patient_info.get("dob", ""),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        image_path,
                        image_bytes,
                        int(results["tumor_present"]),
                        int(results["predicted_grade"]),
                        float(results["grade_confidence"]),
                        float(results["tumor_area"]),
                        json.dumps(serializable),
                    ),
                )
                conn.commit()
                return cur.lastrowid
            except Exception as e:
                print(f"[DB] save_prediction error: {e}")
                raise
            finally:
                if conn:
                    conn.close()

    def get_all_predictions(self):
        conn = None
        try:
            conn = self._connect()
            cur  = conn.cursor()
            # Exclude image_blob from listing (too large)
            cur.execute("""SELECT id, patient_name, patient_id, phone, gender, dob,
                                  upload_date, image_path,
                                  tumor_present, predicted_grade, grade_confidence,
                                  tumor_area, results_json
                           FROM predictions ORDER BY upload_date DESC""")
            return cur.fetchall()
        except Exception as e:
            print(f"[DB] get_all_predictions error: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_prediction(self, prediction_id):
        conn = None
        try:
            conn = self._connect()
            cur  = conn.cursor()
            cur.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,))
            return cur.fetchone()
        except Exception as e:
            print(f"[DB] get_prediction error: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def search_predictions(self, query):
        """Search by patient name or patient ID."""
        conn = None
        try:
            conn = self._connect()
            cur  = conn.cursor()
            q = f"%{query}%"
            cur.execute("""SELECT id, patient_name, patient_id, phone, gender, dob,
                                  upload_date, image_path,
                                  tumor_present, predicted_grade, grade_confidence,
                                  tumor_area, results_json
                           FROM predictions
                           WHERE patient_name LIKE ? OR patient_id LIKE ?
                           ORDER BY upload_date DESC""", (q, q))
            return cur.fetchall()
        except Exception as e:
            print(f"[DB] search_predictions error: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def delete_prediction(self, prediction_id):
        conn = None
        with _db_lock:
            try:
                pred = self.get_prediction(prediction_id)
                if pred:
                    try:
                        mask_path = json.loads(pred[-1]).get("tumor_mask")
                        if mask_path and os.path.exists(mask_path):
                            os.remove(mask_path)
                    except Exception:
                        pass
                conn = self._connect()
                cur  = conn.cursor()
                cur.execute("DELETE FROM predictions WHERE id=?", (prediction_id,))
                conn.commit()
                return True
            except Exception as e:
                print(f"[DB] delete_prediction error: {e}")
                return False
            finally:
                if conn:
                    conn.close()

    # ════════════════════════════════════════════════════════════════
    # PIPELINE 2 — Clinical
    # ════════════════════════════════════════════════════════════════
    def save_clinical_prediction(self, patient_info, features, result):
        """
        patient_info : dict {name, patient_id, phone, gender, dob}
        features     : dict {idh1, age, pten, egfr, atrx}
        result       : dict returned by VQC2Predictor.predict()
        """
        conn = None
        with _db_lock:
            try:
                conn = self._connect()
                cur  = conn.cursor()
                cur.execute(
                    """INSERT INTO clinical_predictions
                       (patient_name, patient_id, phone, gender, dob,
                        upload_date,
                        idh1, age, pten, egfr, atrx,
                        predicted_class, confidence, raw_output,
                        lgg_probability, gbm_probability)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        patient_info.get("name", ""),
                        patient_info.get("patient_id", ""),
                        patient_info.get("phone", ""),
                        patient_info.get("gender", ""),
                        patient_info.get("dob", ""),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        int(features["idh1"]),
                        float(features["age"]),
                        int(features["pten"]),
                        int(features["egfr"]),
                        int(features["atrx"]),
                        result["predicted_class"],
                        float(result["confidence"]),
                        float(result["raw_output"]),
                        float(result["lgg_probability"]),
                        float(result["gbm_probability"]),
                    ),
                )
                conn.commit()
                return cur.lastrowid
            except Exception as e:
                print(f"[DB] save_clinical_prediction error: {e}")
                raise
            finally:
                if conn:
                    conn.close()

    def get_all_clinical_predictions(self):
        conn = None
        try:
            conn = self._connect()
            cur  = conn.cursor()
            cur.execute(
                "SELECT * FROM clinical_predictions ORDER BY upload_date DESC"
            )
            return cur.fetchall()
        except Exception as e:
            print(f"[DB] get_all_clinical_predictions error: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def search_clinical_predictions(self, query):
        """Search by patient name or patient ID."""
        conn = None
        try:
            conn = self._connect()
            cur  = conn.cursor()
            q = f"%{query}%"
            cur.execute("""SELECT * FROM clinical_predictions
                           WHERE patient_name LIKE ? OR patient_id LIKE ?
                           ORDER BY upload_date DESC""", (q, q))
            return cur.fetchall()
        except Exception as e:
            print(f"[DB] search_clinical_predictions error: {e}")
            return []
        finally:
            if conn:
                conn.close()

    # ════════════════════════════════════════════════════════════════
    # STATS
    # ════════════════════════════════════════════════════════════════
    def get_statistics(self):
        conn = None
        try:
            conn = self._connect()
            cur  = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM predictions")
            total_mri = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM predictions WHERE tumor_present=1")
            tumor_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM clinical_predictions")
            total_clin = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM clinical_predictions WHERE predicted_class='GBM'")
            gbm_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM clinical_predictions WHERE predicted_class='LGG'")
            lgg_count = cur.fetchone()[0]
            return {
                "total_mri":   total_mri,
                "tumor_count": tumor_count,
                "total_clin":  total_clin,
                "gbm_count":   gbm_count,
                "lgg_count":   lgg_count,
            }
        except Exception as e:
            print(f"[DB] get_statistics error: {e}")
            return {}
        finally:
            if conn:
                conn.close()
