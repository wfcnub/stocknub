"""Read technical CSVs and metadata, without selecting features or sessions."""
import csv
import hashlib
import io
import json
import logging
from contextlib import contextmanager

import pandas as pd

from app.errors.technical import (
    FeatureFilterUnavailable, InvalidModelMetadata, InvalidTechnicalData,
    TechnicalDataMissing,
)
from utils import paths

logger = logging.getLogger(__name__)


def _reject_json_constant(value):
    raise ValueError("Nonstandard JSON constant")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


class TechnicalRepository:
    def get_technical_indicators(self, ticker: str) -> pd.DataFrame:
        path = paths.get_technical_path(ticker)
        try:
            content = path.read_text(encoding="utf-8-sig")
        except FileNotFoundError as exc:
            raise TechnicalDataMissing() from exc
        except (OSError, UnicodeError) as exc:
            raise InvalidTechnicalData() from exc
        try:
            # pandas mangles duplicate headers and can tolerate short rows.
            # Validate the actual CSV before letting pandas infer numeric types.
            rows = csv.reader(io.StringIO(content), strict=True)
            headers = next(rows)
            if (not headers or len(headers) != len(set(headers)) or "Date" not in headers
                    or any(not name or name != name.strip() for name in headers)):
                raise ValueError("Invalid technical headers")
            row_count = 0
            for row in rows:
                if len(row) != len(headers):
                    raise ValueError("Malformed technical row")
                row_count += 1
            if not row_count:
                raise ValueError("Empty technical CSV")
            return pd.read_csv(io.StringIO(content), float_precision="round_trip")
        except (StopIteration, ValueError, csv.Error, pd.errors.ParserError) as exc:
            logger.warning("invalid_technical_data ticker=%s path=%s", ticker, path)
            raise InvalidTechnicalData() from exc

    @staticmethod
    def _metadata_path(ticker: str, label_type: str, window: str):
        return paths.get_model_artifact_path(1, label_type, ticker, window, "metadata", "json")

    @staticmethod
    def _revision(path):
        try:
            stat = path.stat()
            return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        except FileNotFoundError as exc:
            raise FeatureFilterUnavailable() from exc
        except OSError as exc:
            raise InvalidModelMetadata() from exc

    @contextmanager
    def metadata_snapshot(self, ticker: str, sources):
        """Reject a bundle changed while reading; no persistent metadata cache.

        Publishers must quiesce serving for multi-file refreshes. File stats
        cannot identify a partially published bundle that predates the request.
        """
        source_paths = [self._metadata_path(ticker, label, window) for label, window in sources]
        before = [self._revision(path) for path in source_paths]
        yield
        if before != [self._revision(path) for path in source_paths]:
            raise FeatureFilterUnavailable()

    def get_model_metadata(self, ticker: str, label_type: str, window: str):
        path = self._metadata_path(ticker, label_type, window)
        try:
            payload = path.read_bytes()
        except FileNotFoundError as exc:
            raise FeatureFilterUnavailable() from exc
        except OSError as exc:
            raise InvalidModelMetadata() from exc
        logger.info("technical_metadata ticker=%s source=%s sha256=%s revision=%s",
                    ticker, path, hashlib.sha256(payload).hexdigest(), self._revision(path))
        try:
            return json.loads(payload, parse_constant=_reject_json_constant,
                              object_pairs_hook=_unique_json_object)
        except (ValueError, UnicodeError) as exc:
            raise InvalidModelMetadata() from exc
