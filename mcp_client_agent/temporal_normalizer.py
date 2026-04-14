"""Temporal Normalizer — pure function for rewriting relative time to absolute dates.

Vendored from workshop/libs/text-ops/text_ops/temporal.py (upstream, maintained).
Self-contained: zero external deps, stdlib only.

Public API:
    normalize_temporal(text: str, ref: datetime) -> str
        Replace relative temporal expressions with absolute ISO dates.
        Example: "上週的手術" + ref=2026-04-13 → "2026-04-06 的手術"

10-pass architecture:
  Pass 0:   Simplified Chinese → Traditional Chinese (temporal chars only)
  Pass 0.5: Chinese number → Arabic (八天→8天)
  Pass 1:   Special day keywords (今天, 昨天, 大後天, yesterday, …)
  Pass 2:   Prefix + weekday (上週一, 下禮拜五, last Monday, …)
  Pass 3:   N units ago/later (3天前, 2週後, 5 days ago, …)
  Pass 4:   Relative period (上個月, 下週, 去年, last week, …)
  Pass 5:   Month + day combo (上個月3號, 下個月15日, …)
  Pass 5.5: Year + month combo (去年三月, 明年十二月, …)
  Pass 6:   Boundary keywords (月底, 年底, 上半年, 上一季, …)
  Pass 7:   Double relative (上上週, 前年, 後年, …)

IMPORTANT: longer patterns run BEFORE shorter ones to avoid
partial matches ("上上週" must not be consumed by "上週").
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# ======================== Chinese number preprocessing ========================

ZH_NUM_MAP: dict[str, str] = {
    "零": "0", "一": "1", "二": "2", "兩": "2", "两": "2",
    "三": "3", "四": "4", "五": "5", "六": "6", "七": "7",
    "八": "8", "九": "9", "十": "10", "百": "00",
    "千": "000", "萬": "0000", "万": "0000",
}

_ZH_SIMPLE_NUM = re.compile(
    r"([零一二兩两三四五六七八九十百千萬万]+)"
    r"(?=[天週周個个月年小時时分秒鐘钟次塊块元])"
)


def _zh_num_to_arabic(zh: str) -> str:
    if not zh:
        return zh
    if len(zh) == 1 and zh in ZH_NUM_MAP:
        return ZH_NUM_MAP[zh]
    result = 0
    current = 0
    for char in zh:
        if char in ("十",):
            if current == 0:
                current = 1
            result += current * 10
            current = 0
        elif char in ("百",):
            if current == 0:
                current = 1
            result += current * 100
            current = 0
        elif char in ("千",):
            if current == 0:
                current = 1
            result += current * 1000
            current = 0
        elif char in ("萬", "万"):
            if current == 0:
                current = 1
            result = (result + current) * 10000
            current = 0
        else:
            digit = ZH_NUM_MAP.get(char)
            if digit and digit.isdigit():
                current = int(digit)
            else:
                return zh
    result += current
    return str(result) if result > 0 else zh


def _preprocess_chinese(text: str) -> str:
    """Convert Chinese numbers to Arabic before temporal unit words."""

    def _repl(m: re.Match[str]) -> str:
        return _zh_num_to_arabic(m.group(1))

    return _ZH_SIMPLE_NUM.sub(_repl, text)


# ======================== Minimal change-tracking stub ========================


@dataclass
class _NormChange:
    op: str
    original_fragment: str
    normalized_fragment: str


# ======================== Simplified→Traditional for temporal keywords ========================

_S2T = str.maketrans(
    {
        "周": "週",
        "个": "個",
        "后": "後",
        "点": "點",
        "时": "時",
        "钟": "鐘",
        "礼": "禮",
        "这": "這",
    }
)

# ======================== Weekday lookup tables ========================

WEEKDAY_MAP: dict[str, int] = {
    "週一": 1, "周一": 1, "星期一": 1, "禮拜一": 1, "礼拜一": 1,
    "週二": 2, "周二": 2, "星期二": 2, "禮拜二": 2, "礼拜二": 2,
    "週三": 3, "周三": 3, "星期三": 3, "禮拜三": 3, "礼拜三": 3,
    "週四": 4, "周四": 4, "星期四": 4, "禮拜四": 4, "礼拜四": 4,
    "週五": 5, "周五": 5, "星期五": 5, "禮拜五": 5, "礼拜五": 5,
    "週六": 6, "周六": 6, "星期六": 6, "禮拜六": 6, "礼拜六": 6,
    "週日": 7, "周日": 7, "星期日": 7, "星期天": 7, "週天": 7,
    "周天": 7, "禮拜日": 7, "禮拜天": 7, "礼拜日": 7, "礼拜天": 7,
}

SPECIAL_DAY_SWIFT: dict[str, int] = {
    "the day before yesterday": -2,
    "the day after tomorrow": 2,
    "yesterday": -1,
    "tomorrow": 1,
    "today": 0,
    "大後天": 3,
    "大前天": -3,
    "大后天": 3,
    "後天": 2,
    "前天": -2,
    "后天": 2,
    "明天": 1,
    "明日": 1,
    "今天": 0,
    "今日": 0,
    "昨天": -1,
    "昨日": -1,
}

# ======================== Weekday helpers ========================


def _this_weekday(ref: datetime, wd: int) -> datetime:
    return ref + timedelta(days=wd - ref.isoweekday())


def _next_weekday(ref: datetime, wd: int) -> datetime:
    return _this_weekday(ref, wd) + timedelta(weeks=1)


def _last_weekday(ref: datetime, wd: int) -> datetime:
    return _this_weekday(ref, wd) - timedelta(weeks=1)


_WD_ALTS = sorted(WEEKDAY_MAP.keys(), key=len, reverse=True)
_WD_PATTERN = "(?:" + "|".join(re.escape(k) for k in _WD_ALTS) + ")"


def _fmt_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _fmt_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _extract_weekday(matched: str) -> str | None:
    for key in sorted(WEEKDAY_MAP.keys(), key=len, reverse=True):
        if key in matched:
            return key
    return None


# ======================== TemporalNormalizer (internal class) ========================


class _TemporalNormalizer:
    """Internal class preserving upstream 7-pass regex pipeline."""

    # ---- compiled patterns (class-level) ----

    # Pass 2: prefix + weekday (Chinese)
    _P2_LAST = re.compile(r"(上一?[個个]?|上)(的)?" + _WD_PATTERN)
    _P2_NEXT = re.compile(r"(下一?[個个]?|下)(的)?" + _WD_PATTERN)
    _P2_THIS = re.compile(r"(這一?[個个]?|這|本)(的)?" + _WD_PATTERN)
    # Pass 2: English
    _EN_WEEKDAYS = r"(?P<ewd>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    _P2_EN_LAST = re.compile(r"\blast\s+" + _EN_WEEKDAYS + r"\b", re.IGNORECASE)
    _P2_EN_NEXT = re.compile(r"\bnext\s+" + _EN_WEEKDAYS + r"\b", re.IGNORECASE)
    _P2_EN_THIS = re.compile(r"\bthis\s+" + _EN_WEEKDAYS + r"\b", re.IGNORECASE)

    # Pass 3: N units ago/later — Chinese
    _DIR_SUFFIX = r"(?:之?[前後后]|以[前後后])"
    _P3_DAYS_ZH = re.compile(r"(\d+)\s*天(" + _DIR_SUFFIX + r")")
    _P3_WEEKS_ZH = re.compile(r"(\d+)\s*(?:[週周]|個?(?:星期|禮拜))(" + _DIR_SUFFIX + r")")
    _P3_MONTHS_ZH = re.compile(r"(\d+)\s*[個个]月(" + _DIR_SUFFIX + r")")
    _P3_YEARS_ZH = re.compile(r"(\d+)\s*年(" + _DIR_SUFFIX + r")")
    _P3_HOURS_ZH = re.compile(r"(\d+)\s*小[時时](" + _DIR_SUFFIX + r")")
    _P3_MINUTES_ZH = re.compile(r"(\d+)\s*分[鐘钟](" + _DIR_SUFFIX + r")")
    _P3_SECONDS_ZH = re.compile(r"(\d+)\s*秒(" + _DIR_SUFFIX + r")")
    _P3_HALF_YEAR = re.compile(r"半年(" + _DIR_SUFFIX + r")")
    _P3_YEAR_HALF = re.compile(r"(\d+)年半(" + _DIR_SUFFIX + r")")
    # English
    _P3_DAYS_EN_AGO = re.compile(r"\b(\d+)\s*days?\s*ago\b", re.IGNORECASE)
    _P3_WEEKS_EN_AGO = re.compile(r"\b(\d+)\s*weeks?\s*ago\b", re.IGNORECASE)
    _P3_MONTHS_EN_AGO = re.compile(r"\b(\d+)\s*months?\s*ago\b", re.IGNORECASE)
    _P3_HOURS_EN_AGO = re.compile(r"\b(\d+)\s*hours?\s*ago\b", re.IGNORECASE)
    _P3_IN_DAYS_EN = re.compile(r"\bin\s+(\d+)\s*days?\b", re.IGNORECASE)

    # Pass 4: relative period
    _P4_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
        (re.compile(r"上[個个]?月"), -30, "date"),
        (re.compile(r"下[個个]?月"), 30, "date"),
        (re.compile(r"上[週周]"), -7, "date"),
        (re.compile(r"下[週周]"), 7, "date"),
        (re.compile(r"去年"), -365, "date"),
        (re.compile(r"明年"), 365, "date"),
        (re.compile(r"今年"), 0, "year_start"),
        (re.compile(r"本[月]"), 0, "month_start"),
        (re.compile(r"本[週周]"), 0, "date"),
        (re.compile(r"\blast\s+week\b", re.IGNORECASE), -7, "date"),
        (re.compile(r"\bnext\s+week\b", re.IGNORECASE), 7, "date"),
        (re.compile(r"\blast\s+month\b", re.IGNORECASE), -30, "date"),
        (re.compile(r"\bnext\s+month\b", re.IGNORECASE), 30, "date"),
        (re.compile(r"\blast\s+year\b", re.IGNORECASE), -365, "date"),
        (re.compile(r"\bnext\s+year\b", re.IGNORECASE), 365, "date"),
    ]

    # Pass 5: month + specific day combo
    _P5_PATTERN = re.compile(r"(上|下|這|本)[個个]?月(\d{1,2})[號号日]?")

    # Pass 6: boundary keywords
    _P6_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"(?:[本這这])?月底"), "month_end"),
        (re.compile(r"(?:[本這这今])?年底"), "year_end"),
        (re.compile(r"(?:[本這这])?月初"), "month_start"),
        (re.compile(r"(?:[本這这今])?年初"), "year_start"),
        (re.compile(r"上半年"), "first_half"),
        (re.compile(r"下半年"), "second_half"),
        (re.compile(r"上一?季"), "last_quarter"),
        (re.compile(r"下一?季"), "next_quarter"),
        (re.compile(r"這一?季|本季"), "this_quarter"),
    ]
    _P6_YEAR_BOUNDARY = re.compile(r"(去年|今年|明年|前年)(年底|年初)")

    # Pass 7: double relative (run BEFORE pass 4)
    _WEEK_UNIT = r"(?:[週周]|星期|禮拜)"
    _P7_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
        (re.compile(r"上上[個个]?" + _WEEK_UNIT), -14, "date"),
        (re.compile(r"下下[個个]?" + _WEEK_UNIT), 14, "date"),
        (re.compile(r"上上[個个]?月"), -60, "date"),
        (re.compile(r"下下[個个]?月"), 60, "date"),
        (re.compile(r"前年"), -730, "date"),
        (re.compile(r"[後后]年"), 730, "date"),
    ]

    # Pass 5.5: year + month combo
    _MONTH_ZH_MAP: dict[str, int] = {
        "一月": 1, "二月": 2, "三月": 3, "四月": 4, "五月": 5, "六月": 6,
        "七月": 7, "八月": 8, "九月": 9, "十月": 10, "十一月": 11, "十二月": 12,
        "1月": 1, "2月": 2, "3月": 3, "4月": 4, "5月": 5, "6月": 6,
        "7月": 7, "8月": 8, "9月": 9, "10月": 10, "11月": 11, "12月": 12,
    }
    _MONTH_ALTS = "|".join(sorted(_MONTH_ZH_MAP.keys(), key=len, reverse=True))
    _P55_YEAR_MONTH = re.compile(r"(前年|去年|今年|明年|[後后]年)(?:的)?(" + _MONTH_ALTS + r")")

    def normalize(self, content: str, ref: datetime) -> tuple[str, list[_NormChange]]:
        changes: list[_NormChange] = []

        # ---- Pass 0: Simplified→Traditional ----
        normalised = content.translate(_S2T)

        # ---- Pass 0.5: Chinese number → Arabic ----
        normalised = _preprocess_chinese(normalised)

        # ---- Pass 0.7: Week synonym normalization ----
        # "上禮拜"/"下禮拜"/"這禮拜"/"本禮拜" (standalone) → 上週/下週/這週/本週
        # Skip when followed by weekday char (一二三四五六日天) to keep pass-2 patterns working.
        normalised = re.sub(
            r"([上下這本這])禮拜(?![一二三四五六日天])",
            r"\1週",
            normalised,
        )

        # ---- Pass 7: double relative (before pass 4 to prevent partial match) ----
        normalised = self._pass7(normalised, ref, changes)

        # ---- Pass 1: special day keywords ----
        normalised = self._pass1(normalised, ref, changes)

        # ---- Pass 2: prefix + weekday ----
        normalised = self._pass2(normalised, ref, changes)

        # ---- Pass 3: N units ago/later ----
        normalised = self._pass3(normalised, ref, changes)

        # ---- Pass 6: boundary keywords ----
        normalised = self._pass6(normalised, ref, changes)

        # ---- Pass 5.5: year + month combo ----
        normalised = self._pass55(normalised, ref, changes)

        # ---- Pass 5: month + specific day ----
        normalised = self._pass5(normalised, ref, changes)

        # ---- Pass 4: relative period ----
        normalised = self._pass4(normalised, ref, changes)

        return normalised, changes

    # ---- pass implementations (copied verbatim from upstream) ----

    def _pass1(self, text: str, ref: datetime, changes: list[_NormChange]) -> str:
        for key, swift in SPECIAL_DAY_SWIFT.items():
            if key not in text.lower() if key.isascii() else key not in text:
                continue
            target = _fmt_date(ref + timedelta(days=swift))

            def _repl(m: re.Match[str], t: str = target, k: str = key) -> str:
                changes.append(_NormChange("temporal", k, t))
                return t

            if key.isascii():
                pattern = r"\b" + re.escape(key) + r"\b"
                text = re.sub(pattern, _repl, text, flags=re.IGNORECASE)
            else:
                text = re.sub(re.escape(key), _repl, text)
        return text

    def _pass2(self, text: str, ref: datetime, changes: list[_NormChange]) -> str:
        def _make_repl(calc_fn):
            def _repl(m: re.Match[str]) -> str:
                wd_str = _extract_weekday(m.group(0))
                if wd_str is None:
                    return m.group(0)
                wd = WEEKDAY_MAP[wd_str]
                dt = calc_fn(ref, wd)
                target = _fmt_date(dt)
                changes.append(_NormChange("temporal", m.group(0), target))
                return target

            return _repl

        text = self._P2_LAST.sub(_make_repl(_last_weekday), text)
        text = self._P2_NEXT.sub(_make_repl(_next_weekday), text)
        text = self._P2_THIS.sub(_make_repl(_this_weekday), text)

        _EN_WD = {
            "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
            "friday": 5, "saturday": 6, "sunday": 7,
        }

        def _en_wd_repl(calc_fn):
            def _repl(m: re.Match[str]) -> str:
                wd = _EN_WD.get(m.group("ewd").lower())
                if wd is None:
                    return m.group(0)
                dt = calc_fn(ref, wd)
                target = _fmt_date(dt)
                changes.append(_NormChange("temporal", m.group(0), target))
                return target

            return _repl

        text = self._P2_EN_LAST.sub(_en_wd_repl(_last_weekday), text)
        text = self._P2_EN_NEXT.sub(_en_wd_repl(_next_weekday), text)
        text = self._P2_EN_THIS.sub(_en_wd_repl(_this_weekday), text)
        return text

    def _pass3(self, text: str, ref: datetime, changes: list[_NormChange]) -> str:
        def _zh_dir(d: str) -> int:
            return -1 if "前" in d else 1

        def _repl_factory(unit: str, is_dt: bool = False):
            def _repl(m: re.Match[str]) -> str:
                n = int(m.group(1))
                sign = _zh_dir(m.group(2))
                if unit == "day":
                    dt = ref + timedelta(days=sign * n)
                elif unit == "week":
                    dt = ref + timedelta(weeks=sign * n)
                elif unit == "month":
                    dt = ref + timedelta(days=sign * n * 30)
                elif unit == "year":
                    dt = ref + timedelta(days=sign * n * 365)
                elif unit == "hour":
                    dt = ref + timedelta(hours=sign * n)
                    target = _fmt_datetime(dt)
                    changes.append(_NormChange("temporal", m.group(0), target))
                    return target
                elif unit == "minute":
                    dt = ref + timedelta(minutes=sign * n)
                    target = _fmt_datetime(dt)
                    changes.append(_NormChange("temporal", m.group(0), target))
                    return target
                elif unit == "second":
                    dt = ref + timedelta(seconds=sign * n)
                    target = _fmt_datetime(dt)
                    changes.append(_NormChange("temporal", m.group(0), target))
                    return target
                else:
                    return m.group(0)
                target = _fmt_datetime(dt) if is_dt else _fmt_date(dt)
                changes.append(_NormChange("temporal", m.group(0), target))
                return target

            return _repl

        def _year_half_repl(m: re.Match[str]) -> str:
            n = int(m.group(1))
            sign = _zh_dir(m.group(2))
            dt = ref + timedelta(days=sign * (n * 365 + 182))
            target = _fmt_date(dt)
            changes.append(_NormChange("temporal", m.group(0), target))
            return target

        text = self._P3_YEAR_HALF.sub(_year_half_repl, text)

        def _half_year_repl(m: re.Match[str]) -> str:
            sign = _zh_dir(m.group(1))
            dt = ref + timedelta(days=sign * 182)
            target = _fmt_date(dt)
            changes.append(_NormChange("temporal", m.group(0), target))
            return target

        text = self._P3_HALF_YEAR.sub(_half_year_repl, text)

        text = self._P3_DAYS_ZH.sub(_repl_factory("day"), text)
        text = self._P3_WEEKS_ZH.sub(_repl_factory("week"), text)
        text = self._P3_MONTHS_ZH.sub(_repl_factory("month"), text)
        text = self._P3_YEARS_ZH.sub(_repl_factory("year"), text)
        text = self._P3_HOURS_ZH.sub(_repl_factory("hour"), text)
        text = self._P3_MINUTES_ZH.sub(_repl_factory("minute"), text)
        text = self._P3_SECONDS_ZH.sub(_repl_factory("second"), text)

        def _en_ago_factory(unit: str):
            def _repl(m: re.Match[str]) -> str:
                n = int(m.group(1))
                if unit == "day":
                    dt = ref - timedelta(days=n)
                elif unit == "week":
                    dt = ref - timedelta(weeks=n)
                elif unit == "month":
                    dt = ref - timedelta(days=n * 30)
                elif unit == "hour":
                    dt = ref - timedelta(hours=n)
                    target = _fmt_datetime(dt)
                    changes.append(_NormChange("temporal", m.group(0), target))
                    return target
                else:
                    return m.group(0)
                target = _fmt_date(dt)
                changes.append(_NormChange("temporal", m.group(0), target))
                return target

            return _repl

        text = self._P3_DAYS_EN_AGO.sub(_en_ago_factory("day"), text)
        text = self._P3_WEEKS_EN_AGO.sub(_en_ago_factory("week"), text)
        text = self._P3_MONTHS_EN_AGO.sub(_en_ago_factory("month"), text)
        text = self._P3_HOURS_EN_AGO.sub(_en_ago_factory("hour"), text)

        def _in_days_repl(m: re.Match[str]) -> str:
            n = int(m.group(1))
            dt = ref + timedelta(days=n)
            target = _fmt_date(dt)
            changes.append(_NormChange("temporal", m.group(0), target))
            return target

        text = self._P3_IN_DAYS_EN.sub(_in_days_repl, text)
        return text

    def _pass4(self, text: str, ref: datetime, changes: list[_NormChange]) -> str:
        for pat, offset, kind in self._P4_PATTERNS:
            if kind == "year_start":
                target = ref.replace(month=1, day=1).strftime("%Y-%m-%d")
            elif kind == "month_start":
                target = ref.replace(day=1).strftime("%Y-%m-%d")
            else:
                target = _fmt_date(ref + timedelta(days=offset))

            def _repl(m: re.Match[str], t: str = target) -> str:
                changes.append(_NormChange("temporal", m.group(0), t))
                return t

            text = pat.sub(_repl, text)
        return text

    def _pass5(self, text: str, ref: datetime, changes: list[_NormChange]) -> str:
        def _repl(m: re.Match[str]) -> str:
            prefix = m.group(1)
            day = int(m.group(2))
            if prefix in ("上",):
                month = ref.month - 1 if ref.month > 1 else 12
                year = ref.year if ref.month > 1 else ref.year - 1
            elif prefix in ("下",):
                month = ref.month + 1 if ref.month < 12 else 1
                year = ref.year if ref.month < 12 else ref.year + 1
            else:
                month = ref.month
                year = ref.year
            max_day = calendar.monthrange(year, month)[1]
            day = min(day, max_day)
            try:
                target = datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                return m.group(0)
            changes.append(_NormChange("temporal", m.group(0), target))
            return target

        return self._P5_PATTERN.sub(_repl, text)

    def _pass6(self, text: str, ref: datetime, changes: list[_NormChange]) -> str:
        def _year_boundary_repl(m: re.Match[str]) -> str:
            year_word, boundary = m.group(1), m.group(2)
            year_offsets = {"前年": -2, "去年": -1, "今年": 0, "明年": 1}
            year = ref.year + year_offsets.get(year_word, 0)
            if boundary == "年底":
                target = f"{year}-12-31"
            else:
                target = f"{year}-01-01"
            changes.append(_NormChange("temporal", m.group(0), target))
            return target

        text = self._P6_YEAR_BOUNDARY.sub(_year_boundary_repl, text)

        def _quarter_start(year: int, q: int) -> datetime:
            return datetime(year, (q - 1) * 3 + 1, 1)

        cur_quarter = (ref.month - 1) // 3 + 1

        for pat, kind in self._P6_PATTERNS:
            if kind == "month_end":
                last_day = calendar.monthrange(ref.year, ref.month)[1]
                target = ref.replace(day=last_day).strftime("%Y-%m-%d")
            elif kind == "year_end":
                target = ref.replace(month=12, day=31).strftime("%Y-%m-%d")
            elif kind == "month_start":
                target = ref.replace(day=1).strftime("%Y-%m-%d")
            elif kind == "year_start":
                target = ref.replace(month=1, day=1).strftime("%Y-%m-%d")
            elif kind == "first_half":
                target = ref.replace(month=1, day=1).strftime("%Y-%m-%d")
            elif kind == "second_half":
                target = ref.replace(month=7, day=1).strftime("%Y-%m-%d")
            elif kind == "last_quarter":
                q = cur_quarter - 1 if cur_quarter > 1 else 4
                y = ref.year if cur_quarter > 1 else ref.year - 1
                target = _fmt_date(_quarter_start(y, q))
            elif kind == "next_quarter":
                q = cur_quarter + 1 if cur_quarter < 4 else 1
                y = ref.year if cur_quarter < 4 else ref.year + 1
                target = _fmt_date(_quarter_start(y, q))
            elif kind == "this_quarter":
                target = _fmt_date(_quarter_start(ref.year, cur_quarter))
            else:
                target = _fmt_date(ref)

            def _repl(m: re.Match[str], t: str = target) -> str:
                changes.append(_NormChange("temporal", m.group(0), t))
                return t

            text = pat.sub(_repl, text)
        return text

    def _pass55(self, text: str, ref: datetime, changes: list[_NormChange]) -> str:
        year_offsets = {"前年": -2, "去年": -1, "今年": 0, "明年": 1, "後年": 2}

        def _repl(m: re.Match[str]) -> str:
            year_word, month_str = m.group(1), m.group(2)
            year = ref.year + year_offsets.get(year_word, 0)
            month = self._MONTH_ZH_MAP.get(month_str)
            if month is None:
                return m.group(0)
            target = f"{year}-{month:02d}-01"
            changes.append(_NormChange("temporal", m.group(0), target))
            return target

        return self._P55_YEAR_MONTH.sub(_repl, text)

    def _pass7(self, text: str, ref: datetime, changes: list[_NormChange]) -> str:
        for pat, offset, _kind in self._P7_PATTERNS:
            target = _fmt_date(ref + timedelta(days=offset))

            def _repl(m: re.Match[str], t: str = target) -> str:
                changes.append(_NormChange("temporal", m.group(0), t))
                return t

            text = pat.sub(_repl, text)
        return text


# ======================== Range helpers ========================


def _week_range(ref: datetime, offset_weeks: int) -> tuple[datetime, datetime]:
    """Return (Monday, Sunday) of the week that is offset_weeks away from ref."""
    this_monday = ref - timedelta(days=ref.isoweekday() - 1)
    start = this_monday + timedelta(weeks=offset_weeks)
    end = start + timedelta(days=6)
    return start, end


def _month_range(ref: datetime, offset_months: int) -> tuple[datetime, datetime]:
    """Return (1st, last day) of the month offset_months away from ref's month."""
    year = ref.year
    month = ref.month + offset_months
    while month < 1:
        month += 12
        year -= 1
    while month > 12:
        month -= 12
        year += 1
    last_day = calendar.monthrange(year, month)[1]
    return datetime(year, month, 1), datetime(year, month, last_day)


def _year_range(ref: datetime, offset_years: int) -> tuple[datetime, datetime]:
    year = ref.year + offset_years
    return datetime(year, 1, 1), datetime(year, 12, 31)


def _quarter_range(year: int, q: int) -> tuple[datetime, datetime]:
    """q is 1..4. Returns (1st of q's first month, last day of q's last month)."""
    start_month = (q - 1) * 3 + 1
    end_month = q * 3
    last_day = calendar.monthrange(year, end_month)[1]
    return datetime(year, start_month, 1), datetime(year, end_month, last_day)


def _fmt_range(start: datetime, end: datetime) -> str:
    return f"{start.strftime('%Y-%m-%d')} 到 {end.strftime('%Y-%m-%d')}"


# ======================== Range pre-pass ========================

# Negative lookahead fragments
_NOT_WEEKDAY = r"(?![一二三四五六日天])"
_NOT_DAY_NUM = r"(?!\d+[號号日])"
_NOT_MONTH = r"(?!(?:[一二三四五六七八九十]|十[一二]|1[0-2]|[1-9])月)"


def _range_prepass(text: str, ref: datetime) -> str:
    """Replace range-valued period expressions with 'YYYY-MM-DD 到 YYYY-MM-DD'.

    Order matters: longer patterns and more-specific patterns must run first.
    After this pass, remaining single-date expressions can be normalized by
    _TemporalNormalizer.normalize() without conflict.
    """

    cur_q = (ref.month - 1) // 3 + 1
    year_off_map = {"前年": -2, "去年": -1, "今年": 0, "明年": 1, "後年": 2, "后年": 2}

    # ---- Year + month combo FIRST (前年三月, 去年12月) — must beat plain 去年 ----
    month_alts = _TemporalNormalizer._MONTH_ALTS

    def _year_month_repl(m: re.Match[str]) -> str:
        year_word = m.group(1)
        month_str = m.group(2)
        y = ref.year + year_off_map.get(year_word, 0)
        mn = _TemporalNormalizer._MONTH_ZH_MAP.get(month_str)
        if mn is None:
            return m.group(0)
        last_day = calendar.monthrange(y, mn)[1]
        return _fmt_range(datetime(y, mn, 1), datetime(y, mn, last_day))

    text = re.sub(
        r"(前年|去年|今年|明年|[後后]年)(?:的)?(" + month_alts + r")",
        _year_month_repl,
        text,
    )

    # ---- Double-relative week (上上週, 下下禮拜) — before single week ----
    _WEEK = r"(?:[週周]|星期|禮拜)"
    text = re.sub(
        r"上上[個个]?" + _WEEK + _NOT_WEEKDAY,
        lambda m: _fmt_range(*_week_range(ref, -2)),
        text,
    )
    text = re.sub(
        r"下下[個个]?" + _WEEK + _NOT_WEEKDAY,
        lambda m: _fmt_range(*_week_range(ref, 2)),
        text,
    )

    # ---- Double-relative month (上上月, 下下個月) — before single month ----
    text = re.sub(
        r"上上[個个]?月" + _NOT_DAY_NUM,
        lambda m: _fmt_range(*_month_range(ref, -2)),
        text,
    )
    text = re.sub(
        r"下下[個个]?月" + _NOT_DAY_NUM,
        lambda m: _fmt_range(*_month_range(ref, 2)),
        text,
    )

    # ---- 最近 N units (last N <unit>, count-based range ending today) ----
    # "最近3天" = ref - (n-1) days → ref
    def _recent_days(m: re.Match[str]) -> str:
        n = max(int(m.group(1)), 1)
        start = ref - timedelta(days=n - 1)
        return _fmt_range(start, ref)

    text = re.sub(r"最近(\d+)\s*天", _recent_days, text)

    def _recent_weeks(m: re.Match[str]) -> str:
        n = max(int(m.group(1)), 1)
        start = ref - timedelta(weeks=n)
        return _fmt_range(start, ref)

    text = re.sub(r"最近(\d+)\s*(?:" + _WEEK + r")", _recent_weeks, text)

    def _recent_months(m: re.Match[str]) -> str:
        n = max(int(m.group(1)), 1)
        start = ref - timedelta(days=n * 30)
        return _fmt_range(start, ref)

    text = re.sub(r"最近(\d+)\s*[個个]?月", _recent_months, text)

    def _recent_years(m: re.Match[str]) -> str:
        n = max(int(m.group(1)), 1)
        start = ref - timedelta(days=n * 365)
        return _fmt_range(start, ref)

    text = re.sub(r"最近(\d+)\s*年", _recent_years, text)

    # "最近一週" / "最近一個月" / "最近一年" (explicit 1 unit)
    text = re.sub(
        r"最近一[個个]?(?:" + _WEEK + r")",
        lambda m: _fmt_range(ref - timedelta(weeks=1), ref),
        text,
    )
    text = re.sub(
        r"最近一[個个]?月",
        lambda m: _fmt_range(ref - timedelta(days=30), ref),
        text,
    )
    text = re.sub(
        r"最近一年",
        lambda m: _fmt_range(ref - timedelta(days=365), ref),
        text,
    )

    # ---- Week synonym: 上禮拜 → 上週 (keeps pass 2 / single-date working) ----
    text = re.sub(r"([上下這本])禮拜" + _NOT_WEEKDAY, r"\1週", text)

    # ---- Single-word week (上週/下週/本週/這週) with negative lookahead weekday ----
    text = re.sub(
        r"上[週周]" + _NOT_WEEKDAY,
        lambda m: _fmt_range(*_week_range(ref, -1)),
        text,
    )
    text = re.sub(
        r"下[週周]" + _NOT_WEEKDAY,
        lambda m: _fmt_range(*_week_range(ref, 1)),
        text,
    )
    text = re.sub(
        r"(?:本|這)[週周]" + _NOT_WEEKDAY,
        lambda m: _fmt_range(*_week_range(ref, 0)),
        text,
    )

    # ---- Single-word month (上月/下月/本月/這月) with negative lookahead day ----
    text = re.sub(
        r"上[個个]?月" + _NOT_DAY_NUM,
        lambda m: _fmt_range(*_month_range(ref, -1)),
        text,
    )
    text = re.sub(
        r"下[個个]?月" + _NOT_DAY_NUM,
        lambda m: _fmt_range(*_month_range(ref, 1)),
        text,
    )
    text = re.sub(
        r"(?:本|這)[個个]?月" + _NOT_DAY_NUM,
        lambda m: _fmt_range(*_month_range(ref, 0)),
        text,
    )

    # ---- Year (去年/今年/明年/前年/後年) — now safe since year+month already consumed ----
    text = re.sub(
        r"前年",
        lambda m: _fmt_range(*_year_range(ref, -2)),
        text,
    )
    text = re.sub(
        r"去年",
        lambda m: _fmt_range(*_year_range(ref, -1)),
        text,
    )
    text = re.sub(
        r"今年",
        lambda m: _fmt_range(*_year_range(ref, 0)),
        text,
    )
    text = re.sub(
        r"明年",
        lambda m: _fmt_range(*_year_range(ref, 1)),
        text,
    )
    text = re.sub(
        r"[後后]年",
        lambda m: _fmt_range(*_year_range(ref, 2)),
        text,
    )

    # ---- Half year / quarter ----
    text = re.sub(
        r"上半年",
        lambda m: _fmt_range(
            datetime(ref.year, 1, 1), datetime(ref.year, 6, 30)
        ),
        text,
    )
    text = re.sub(
        r"下半年",
        lambda m: _fmt_range(
            datetime(ref.year, 7, 1), datetime(ref.year, 12, 31)
        ),
        text,
    )

    def _last_q_repl(m: re.Match[str]) -> str:
        q = cur_q - 1 if cur_q > 1 else 4
        y = ref.year if cur_q > 1 else ref.year - 1
        return _fmt_range(*_quarter_range(y, q))

    def _next_q_repl(m: re.Match[str]) -> str:
        q = cur_q + 1 if cur_q < 4 else 1
        y = ref.year if cur_q < 4 else ref.year + 1
        return _fmt_range(*_quarter_range(y, q))

    def _this_q_repl(m: re.Match[str]) -> str:
        return _fmt_range(*_quarter_range(ref.year, cur_q))

    text = re.sub(r"上一?季", _last_q_repl, text)
    text = re.sub(r"下一?季", _next_q_repl, text)
    text = re.sub(r"(?:這一?季|本季)", _this_q_repl, text)

    return text


# ======================== Public API ========================

_SINGLETON = _TemporalNormalizer()

# Post-processing: pad spaces around ISO dates so downstream regex using \b
# word-boundary can match them even when adjacent to CJK characters (which
# Python 3 re module treats as \w, breaking \b).
_ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Collapse a chain of 3+ ISO dates joined by 到/至 into a single (earliest, latest)
# range. Rationale: "去年一月到今年三月" becomes
#   "2025-01-01 到 2025-01-31 到 2026-03-01 到 2026-03-31"
# after range_prepass (each period expanded to its own range). The downstream
# parser takes only the first two dates, so we must collapse the chain into a
# single outer-bound range before returning.
_ISO_CHAIN_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})(?:\s*(?:到|至)\s*\d{4}-\d{2}-\d{2}){2,}"
)


def _collapse_iso_chains(text: str) -> str:
    """Collapse `A 到 B 到 C 到 D` → `A 到 D` (time-sorted first/last)."""

    def _repl(m: re.Match[str]) -> str:
        chain = m.group(0)
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", chain)
        if not dates:
            return chain
        sorted_dates = sorted(dates)  # ISO YYYY-MM-DD is lexicographic = chronological
        return f"{sorted_dates[0]} 到 {sorted_dates[-1]}"

    return _ISO_CHAIN_RE.sub(_repl, text)


def _pad_iso_dates(text: str) -> str:
    """Ensure each YYYY-MM-DD has whitespace on both sides."""

    def _repl(m: re.Match[str]) -> str:
        start, end = m.span()
        lead = "" if start == 0 or text[start - 1] == " " else " "
        trail = "" if end == len(text) or text[end] == " " else " "
        return f"{lead}{m.group(0)}{trail}"

    return _ISO_DATE_RE.sub(_repl, text)


def normalize_temporal(text: str, ref: datetime | None = None) -> str:
    """Rewrite relative temporal expressions to absolute ISO single dates.

    Period expressions ("上週", "上個月") are treated as anchor points (single
    dates), preserving upstream behavior. For range-aware rewriting, see
    normalize_temporal_range().

    Args:
        text: input text (Chinese or English).
        ref:  reference datetime for "today". Defaults to datetime.now() if None.

    Returns:
        Rewritten text. Never raises; falls back to original on any error.

    Examples:
        >>> from datetime import datetime
        >>> ref = datetime(2026, 4, 13)  # Monday
        >>> normalize_temporal("上週的手術報告", ref)
        '2026-04-06 的手術報告'
        >>> normalize_temporal("三天前開的刀", ref)
        '2026-04-10 開的刀'
    """
    if not text:
        return text
    if ref is None:
        ref = datetime.now()
    try:
        normalised, _changes = _SINGLETON.normalize(text, ref)
        return _pad_iso_dates(normalised)
    except Exception:
        return text


def normalize_temporal_range(text: str, ref: datetime | None = None) -> str:
    """Rewrite relative temporal expressions, expanding periods to full date ranges.

    Behaviour:
    - Period expressions ("上週", "上個月", "去年", "最近3天") become
      "YYYY-MM-DD 到 YYYY-MM-DD" (Traditional Chinese "to").
    - Single-date expressions ("今天", "昨天", "3天前", "上週一") stay as
      single absolute ISO dates (unchanged from normalize_temporal).

    This is the preferred form for LLM tool-calling when the tool takes
    start/end parameters, since LLMs can copy both dates verbatim.

    Args:
        text: input text (Chinese or English).
        ref:  reference datetime for "today". Defaults to datetime.now() if None.

    Returns:
        Rewritten text. Never raises.

    Examples:
        >>> from datetime import datetime
        >>> ref = datetime(2026, 4, 13)  # Monday
        >>> normalize_temporal_range("查上週的手術", ref)
        '查2026-04-06 到 2026-04-12的手術'
        >>> normalize_temporal_range("去年三月有多少", ref)
        '2025-03-01 到 2025-03-31有多少'
        >>> normalize_temporal_range("最近3天開幾刀", ref)
        '2026-04-11 到 2026-04-13開幾刀'
        >>> normalize_temporal_range("3天前那台手術", ref)
        '2026-04-10那台手術'
    """
    if not text:
        return text
    if ref is None:
        ref = datetime.now()
    try:
        # Pass 0 + 0.5: S2T + Chinese number → Arabic (must run before range patterns)
        t = text.translate(_S2T)
        t = _preprocess_chinese(t)

        # Range pre-pass: consume period expressions as ranges
        t = _range_prepass(t, ref)

        # Fall through to single-date normalizer for leftovers
        # (今天/昨天/N天前/N週前/上週一/N年半前 etc.)
        t, _ = _SINGLETON.normalize(t, ref)

        # Collapse "A 到 B 到 C 到 D" chains into single (earliest, latest) range
        # so "去年一月到今年三月" → "2025-01-01 到 2026-03-31" (not split 4 ways).
        t = _collapse_iso_chains(t)

        return _pad_iso_dates(t)
    except Exception:
        return text
