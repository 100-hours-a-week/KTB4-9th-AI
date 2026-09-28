"""rename category HASH_TABLE to HASH

Revision ID: 46cefa2003a5
Revises: 88bb96af4ad1
Create Date: 2026-09-28 00:38:58.819865

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '46cefa2003a5'
down_revision: Union[str, Sequence[str], None] = '88bb96af4ad1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# 백엔드 enum과 같게 HASH_TABLE을 HASH로 바꾼다.
# 카테고리는 문자열 컬럼(native_enum=False)이라 스키마는 그대로 두고 값만 바꾼다.
# 바꾸지 않으면 기존 행을 읽을 때 enum 변환이 실패하고, 중복 검사도 기존 임베딩과 끊긴다.
CATEGORY_COLUMNS = [
    ("generated_problems", "category"),
    ("discarded_problems", "requested_category"),
    ("problem_embeddings", "category"),
    ("fewshot_seeds", "category"),
    ("battle_problems", "category"),
    ("battle_problem_embeddings", "category"),
]


def rename(old: str, new: str) -> None:
    for table, column in CATEGORY_COLUMNS:
        op.execute(
            sa.text(f"UPDATE {table} SET {column} = :new WHERE {column} = :old")
            .bindparams(old=old, new=new)
        )


def upgrade() -> None:
    """Upgrade schema."""
    rename("HASH_TABLE", "HASH")


def downgrade() -> None:
    """Downgrade schema."""
    rename("HASH", "HASH_TABLE")
