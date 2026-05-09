# -*- coding: utf-8 -*-
"""
SQLite数据库管理
"""

import sqlite3
import threading
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import os

from .config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite数据库管理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.db_path = os.path.join(settings.DATA_DIR, "async_tts.db")
        self._local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """获取线程本地数据库连接"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_database(self):
        """初始化数据库表"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            conn = self._get_connection()
            cursor = conn.cursor()

            # 创建异步TTS任务表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS async_tts_tasks (
                    task_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'RUNNING',
                    text TEXT NOT NULL,
                    voice TEXT NOT NULL,
                    sample_rate INTEGER NOT NULL DEFAULT 16000,
                    format TEXT NOT NULL DEFAULT 'wav',
                    enable_subtitle BOOLEAN NOT NULL DEFAULT FALSE,
                    enable_notify BOOLEAN NOT NULL DEFAULT FALSE,
                    notify_url TEXT,
                    audio_address TEXT,
                    sentences TEXT,
                    error_code INTEGER DEFAULT 20000000,
                    error_message TEXT DEFAULT 'RUNNING',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            self._ensure_columns(
                cursor,
                "async_tts_tasks",
                {
                    "prompt": "TEXT DEFAULT ''",
                    "emotion": "TEXT",
                    "emotion_intensity": "REAL",
                    "emotion_source": "TEXT",
                },
            )

            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_status
                ON async_tts_tasks(status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON async_tts_tasks(created_at)
            """)

            # 创建异步ASR长录音任务表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS async_asr_tasks (
                    task_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'RUNNING',
                    audio_address TEXT,
                    format TEXT NOT NULL DEFAULT 'wav',
                    sample_rate INTEGER NOT NULL DEFAULT 16000,
                    vocabulary_id TEXT,
                    hotwords TEXT,
                    customization_id TEXT NOT NULL DEFAULT 'sensevoice-small',
                    enable_punctuation_prediction BOOLEAN NOT NULL DEFAULT FALSE,
                    enable_inverse_text_normalization BOOLEAN NOT NULL DEFAULT FALSE,
                    enable_voice_detection BOOLEAN NOT NULL DEFAULT TRUE,
                    disfluency BOOLEAN NOT NULL DEFAULT FALSE,
                    dolphin_lang_sym TEXT NOT NULL DEFAULT 'zh',
                    dolphin_region_sym TEXT NOT NULL DEFAULT 'SHANGHAI',
                    enable_notify BOOLEAN NOT NULL DEFAULT FALSE,
                    notify_url TEXT,
                    result TEXT,
                    duration_ms INTEGER,
                    error_code INTEGER DEFAULT 20000000,
                    error_message TEXT DEFAULT 'RUNNING',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            self._ensure_columns(
                cursor,
                "async_asr_tasks",
                {
                    "enable_emotion": "BOOLEAN NOT NULL DEFAULT FALSE",
                    "return_rich_text": "BOOLEAN NOT NULL DEFAULT FALSE",
                    "emotion": "TEXT",
                    "emotion_confidence": "REAL",
                    "raw_rich_text": "TEXT",
                    "audio_bytes": "BLOB",
                },
            )

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_async_asr_task_status
                ON async_asr_tasks(status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_async_asr_created_at
                ON async_asr_tasks(created_at)
            """)

            conn.commit()
            logger.info(f"数据库初始化完成: {self.db_path}")

        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise

    def _ensure_columns(self, cursor: sqlite3.Cursor, table: str, columns: Dict[str, str]) -> None:
        """Add columns to existing SQLite tables without requiring a separate migration."""
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        for column, definition in columns.items():
            if column not in existing:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def create_task(self, task_data: Dict[str, Any]) -> bool:
        """创建异步TTS任务"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO async_tts_tasks (
                    task_id, request_id, text, voice, sample_rate,
                    format, enable_subtitle, prompt, emotion, emotion_intensity,
                    emotion_source, enable_notify, notify_url, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_data['task_id'],
                task_data['request_id'],
                task_data['text'],
                task_data['voice'],
                task_data['sample_rate'],
                task_data['format'],
                task_data['enable_subtitle'],
                task_data.get('prompt', ''),
                task_data.get('emotion'),
                task_data.get('emotion_intensity'),
                task_data.get('emotion_source'),
                task_data.get('enable_notify', False),
                task_data.get('notify_url'),
                'RUNNING',
                'RUNNING'
            ))

            conn.commit()
            logger.info(f"创建异步TTS任务: {task_data['task_id']}")
            return True

        except Exception as e:
            logger.error(f"创建任务失败: {e}")
            return False

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM async_tts_tasks WHERE task_id = ?
            """, (task_id,))

            row = cursor.fetchone()
            if row:
                task = dict(row)
                # 解析JSON字段
                if task['sentences']:
                    task['sentences'] = json.loads(task['sentences'])
                return task
            return None

        except Exception as e:
            logger.error(f"获取任务失败: {e}")
            return None

    def update_task_status(self, task_id: str, status: str,
                          audio_address: str = None,
                          sentences: List[Dict] = None,
                          error_code: int = 20000000,
                          error_message: str = "SUCCESS") -> bool:
        """更新任务状态"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            params = [status, error_code, error_message]
            sql_parts = ["status = ?", "error_code = ?", "error_message = ?", "updated_at = CURRENT_TIMESTAMP"]

            if audio_address:
                sql_parts.append("audio_address = ?")
                params.append(audio_address)

            if sentences:
                sql_parts.append("sentences = ?")
                params.append(json.dumps(sentences, ensure_ascii=False))

            if status in ['SUCCESS', 'FAILED']:
                sql_parts.append("completed_at = CURRENT_TIMESTAMP")

            params.append(task_id)

            cursor.execute(f"""
                UPDATE async_tts_tasks
                SET {', '.join(sql_parts)}
                WHERE task_id = ?
            """, params)

            conn.commit()
            logger.info(f"更新任务状态: {task_id} -> {status}")
            return True

        except Exception as e:
            logger.error(f"更新任务状态失败: {e}")
            return False

    def get_pending_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取待处理的任务"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM async_tts_tasks
                WHERE status = 'RUNNING'
                ORDER BY created_at ASC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"获取待处理任务失败: {e}")
            return []

    def cleanup_old_tasks(self, days: int = 7) -> int:
        """清理旧任务（默认7天前的任务）"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cutoff_date = datetime.now() - timedelta(days=days)

            cursor.execute("""
                DELETE FROM async_tts_tasks
                WHERE created_at < ? AND status IN ('SUCCESS', 'FAILED')
            """, (cutoff_date,))

            deleted_count = cursor.rowcount
            conn.commit()

            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 个旧任务")

            return deleted_count

        except Exception as e:
            logger.error(f"清理旧任务失败: {e}")
            return 0

    def create_asr_task(self, task_data: Dict[str, Any]) -> bool:
        """创建异步ASR任务"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO async_asr_tasks (
                    task_id, request_id, audio_address, audio_bytes, format, sample_rate,
                    vocabulary_id, hotwords, customization_id,
                    enable_punctuation_prediction, enable_inverse_text_normalization,
                    enable_voice_detection, disfluency, dolphin_lang_sym,
                    dolphin_region_sym, enable_emotion, return_rich_text,
                    enable_notify, notify_url, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_data["task_id"],
                task_data["request_id"],
                task_data.get("audio_address") or "",
                task_data.get("audio_bytes"),
                task_data["format"],
                task_data["sample_rate"],
                task_data.get("vocabulary_id"),
                task_data.get("hotwords"),
                task_data["customization_id"],
                task_data["enable_punctuation_prediction"],
                task_data["enable_inverse_text_normalization"],
                task_data["enable_voice_detection"],
                task_data["disfluency"],
                task_data["dolphin_lang_sym"],
                task_data["dolphin_region_sym"],
                task_data.get("enable_emotion", False),
                task_data.get("return_rich_text", False),
                task_data.get("enable_notify", False),
                task_data.get("notify_url"),
                "RUNNING",
                "RUNNING",
            ))

            conn.commit()
            logger.info(f"创建异步ASR任务: {task_data['task_id']}")
            return True

        except Exception as e:
            logger.error(f"创建异步ASR任务失败: {e}")
            return False

    def get_asr_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取异步ASR任务信息"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM async_asr_tasks WHERE task_id = ?", (task_id,))

            row = cursor.fetchone()
            return dict(row) if row else None

        except Exception as e:
            logger.error(f"获取异步ASR任务失败: {e}")
            return None

    def get_pending_asr_tasks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取待处理的异步ASR任务"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM async_asr_tasks
                WHERE status = 'RUNNING'
                ORDER BY created_at ASC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"获取待处理异步ASR任务失败: {e}")
            return []

    def update_asr_task_status(
        self,
        task_id: str,
        status: str,
        result: str = None,
        duration_ms: int = None,
        emotion: str = None,
        emotion_confidence: float = None,
        raw_rich_text: str = None,
        error_code: int = 20000000,
        error_message: str = "SUCCESS",
    ) -> bool:
        """更新异步ASR任务状态"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            params = [status, error_code, error_message]
            sql_parts = [
                "status = ?",
                "error_code = ?",
                "error_message = ?",
                "updated_at = CURRENT_TIMESTAMP",
            ]

            if result is not None:
                sql_parts.append("result = ?")
                params.append(result)

            if duration_ms is not None:
                sql_parts.append("duration_ms = ?")
                params.append(duration_ms)

            if emotion is not None:
                sql_parts.append("emotion = ?")
                params.append(emotion)

            if emotion_confidence is not None:
                sql_parts.append("emotion_confidence = ?")
                params.append(emotion_confidence)

            if raw_rich_text is not None:
                sql_parts.append("raw_rich_text = ?")
                params.append(raw_rich_text)

            if status in ["SUCCESS", "FAILED"]:
                sql_parts.append("completed_at = CURRENT_TIMESTAMP")

            params.append(task_id)
            cursor.execute(f"""
                UPDATE async_asr_tasks
                SET {', '.join(sql_parts)}
                WHERE task_id = ?
            """, params)

            conn.commit()
            logger.info(f"更新异步ASR任务状态: {task_id} -> {status}")
            return True

        except Exception as e:
            logger.error(f"更新异步ASR任务状态失败: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')


# 全局数据库管理器实例
db_manager = DatabaseManager()
