from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.sqlite import insert


def upsert_method(table, conn, keys, data_iter):

    meta = MetaData()
    sql_table = Table(table.name, meta)

    data = [dict(zip(keys, row)) for row in data_iter]
    insert_stmt = insert(sql_table).values(data)
    print(insert_stmt.excluded.items())
    print(len(insert_stmt.excluded))
    upsert_stmt = insert_stmt.on_conflict_do_update(set_={x.name: x for x in insert_stmt.excluded})

    result = conn.execute(upsert_stmt)

    return result.rowcount
