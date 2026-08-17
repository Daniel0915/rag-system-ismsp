import fs from "node:fs";
import path from "node:path";
import XLSX from "xlsx";

/**
 * KISA의 "ISMS-P 인증기준 세부점검항목" xlsx는 셀 병합 대신 반복되는 빈 셀로
 * 상위 분류(분야/항목/상세내용)를 표현한다 — 실제 병합 범위는 헤더 행에만 있으므로
 * 데이터 행은 위 값을 그대로 이어받는(forward-fill) 방식으로 파싱해야 한다.
 */

const SRC = "data/isms-p/ISMS-P_인증기준_세부점검항목(2023.10.31).xlsx";
const OUT = "data/isms-criteria.json";

const domainBySheet: Record<string, string> = {
  "1.관리체계 수립 및 운영": "1",
  "2.보호대책 요구사항": "2",
  "3.개인정보 처리단계별 요구사항": "3",
};

interface CriteriaItem {
  code: string;
  category_code: string | null;
  category: string | null;
  title: string | null;
  detail: string | null;
  domain: string;
  checks: string[];
}

function parse(): CriteriaItem[] {
  const wb = XLSX.readFile(path.resolve(SRC));
  const items: CriteriaItem[] = [];

  for (const sheetName of wb.SheetNames) {
    const domain = domainBySheet[sheetName];
    if (!domain) continue;

    const sheet = wb.Sheets[sheetName];
    const rows: unknown[][] = XLSX.utils.sheet_to_json(sheet, { header: 1, raw: false });

    let categoryCode: string | null = null;
    let categoryName: string | null = null;
    let itemCode: string | null = null;
    let itemTitle: string | null = null;
    let detail: string | null = null;
    let current: CriteriaItem | null = null;

    for (let i = 2; i < rows.length; i++) {
      const row = rows[i] ?? [];
      const [c0, c1, c2, c3, c4, c5] = row as (string | null)[];

      if (c0 != null && String(c0).trim() !== "") categoryCode = String(c0).trim();
      if (c1 != null && String(c1).trim() !== "") categoryName = String(c1).trim();
      if (c2 != null && String(c2).trim() !== "") itemCode = String(c2).trim();
      if (c3 != null && String(c3).trim() !== "") itemTitle = String(c3).trim();
      if (c4 != null && String(c4).trim() !== "") detail = String(c4).trim();
      const check = c5 != null ? String(c5).trim() : "";

      if (!itemCode) continue;

      if (!current || current.code !== itemCode) {
        current = {
          code: itemCode,
          category_code: categoryCode,
          category: categoryName,
          title: itemTitle,
          detail,
          domain,
          checks: [],
        };
        items.push(current);
      }
      if (check) current.checks.push(check);
    }
  }

  return items;
}

const items = parse();
fs.writeFileSync(OUT, JSON.stringify(items, null, 2) + "\n", "utf-8");
console.log(`${OUT}: ${items.length}개 항목, ${items.reduce((a, i) => a + i.checks.length, 0)}개 확인사항`);
