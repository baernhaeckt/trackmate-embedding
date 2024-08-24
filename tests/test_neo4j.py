from neo4j import GraphDatabase

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "qaywsxedc")


def add_location(driver, location, next_location, image_embedding_1, image_embedding_2):
    driver.execute_query(
        "MERGE (a:Location {name: $location, embedding: $image_embedding_1}) "
        "MERGE (b:Location {name: $next_location, embedding: $image_embedding_2}) "
        "MERGE (a)-[:GOES]->(b)",
        location=location,
        next_location=next_location,
        image_embedding_1=image_embedding_1,
        image_embedding_2=image_embedding_2,
        database="neo4j",
    )


def get_location(driver, image_embedding):
    return_value = driver.execute_query(
        "WITH $image_embedding AS external_vector "
        "MATCH (a:Location) "
        "WHERE gds.similarity.euclidean(a.embedding, external_vector) > 0.8 "
        "RETURN a, gds.similarity.euclidean(a.embedding, external_vector) AS similarity "
        "ORDER BY similarity DESC",
        image_embedding=image_embedding
    )

    print(return_value)


if __name__ == '__main__':
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        get_location(driver, [0.1, 0.2, 0.3, 0.4])
        # add_location(
        #     driver,
        #     "Eingang",
        #     "Raum 058",
        #     [0.1, 0.2, 0.3, 0.4],
        #     [0.2, 0.3, 0.4, 0.5]
        # )
