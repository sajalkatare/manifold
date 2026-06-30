```sql
docker exec -it rag_pgvector psql -U rag -d ragdb


--List Extensions
\dx

--List Tables
\d

--Display table
SELECT *
FROM langchain_pg_embedding;


--Display Entries

SELECT
    id,
    collection_id,
    document,
    cmetadata
FROM langchain_pg_embedding;

```
