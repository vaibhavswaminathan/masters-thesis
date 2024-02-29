#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

from rdflib import Graph, Namespace, BNode, RDF
import pprint

XSD = Namespace("http://www.w3.org/2001/XMLSchema#")
# g = Graph(bind_namespaces='none')
g = Graph()
g.parse("a_model.ttl")

# print(len(g))

# for itr in g.namespaces():
#     print(itr)

# for stmt in g:
#     pprint.pprint(stmt)

# Query the RDF graph (to extract rules)
knows_query = """
SELECT ?subj1 ?pred1 ?obj1
WHERE {
    ?stmt1 a bldg:Statement ;
        rdf:subject ?subj1 ;
        rdf:predicate ?pred1 ;
        rdf:object ?obj1 .
}"""

blank_query = """
SELECT ?rel ?obj1
WHERE {
    [] ?rel ?obj1 .
}"""

construct_query = """
SELECT ?object 
WHERE {
  ?subject ?predicate ?object .
  FILTER (str(?subject) = "nb49364a469244363afb945e05d55f253b6")
}
"""

subject = "bldg:blank2"
sparql_query = """
    SELECT ?predicate ?object
    WHERE {
    %s ?predicate ?object .
    }
""" % subject

# qres = g.query(blank_query)
# for row in qres:
#     # print(f"object: {row.obj1}")
#     print(row)

# Define a function to recursively query RDF triples within blank nodes
def query_blank_nodes(graph, subject):
    results = graph.query(
        """
        SELECT ?predicate ?object
        WHERE { 
        { %s ?predicate ?object . }
        UNION
        { ?some ?pred %s . }
        }
        """ % (subject, subject))
    
    for row in results:
        predicate, obj = row
        print(f"{subject} {predicate} {obj}")
        
        # If the object is a blank node, recursively query its triples
        if isinstance(obj, BNode):
            print(True)
            query_blank_nodes(graph, obj)

# Start the recursive query from the root node (you may need to adapt this to your specific graph)
root_subject = "bldg:blank2"
query_blank_nodes(g, root_subject)