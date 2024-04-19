import os
import json
from rdflib import Graph, query, Namespace

# Define the path to your Turtle file
file_path = "a_model.ttl"

# Create a Graph object
graph = Graph()

# Read the Turtle file contents into the graph
graph.parse(file_path, format="ttl")

# Define your search parameter (e.g., a specific URI)
predicate_search_param = "https://brickschema.org/schema/Brick#isPointOf"  # Replace with your actual parameter
# subsystem_param = "http://example.com/mybuilding#ODA"
# component_param = "heater"
# substance_param = "valve"
has_point_param = "https://brickschema.org/schema/Brick#hasPoint"
is_pointof_param = "https://brickschema.org/schema/Brick#isPointOf"

# Function to find all triples with a specific subject
def find_triples_by_subject(subject):
  triples = []
  for s, p, o in graph:
    if s.strip() == subject.strip() or p.strip() == subject.strip() or o.strip() == subject.strip():
      triples.append((s, p, o))
  return triples

def find_keyword_in_graph(keyword):
  triples = []
  for s, p, o in graph:
    strings = [s.strip().lower(), p.strip().lower(), o.strip().lower()]
    has_result = any(keyword in string for string in strings)
    if has_result:
      triples.append((s, p, o))
  return triples

def find_iterative_matches(search_param, substance_param):
  """
  Search all triples in graph firstly for:
  1. Component(search_param) / substance and secondly,
  2. brick:hasPoint or brick:isPointOf relations
  Return the subject to use for adding new triple
  """
  key_triples = find_keyword_in_graph(search_param)
  if not key_triples:
    print(f"No triples found having: {search_param}")
    return None
  for s, p, o in key_triples:
      # print(f"\t- ({s}, {p}, {o})")
      # check for brick:hasPoint or brick:isPointOf relations
      res_match = next((string for string in [has_point_param, is_pointof_param] if p.strip() == string), None)
      if res_match != None:
        if res_match == has_point_param:
          # check if object has substance
          subs_match = substance_param in o.strip().lower()
          if subs_match:
            new_subject = o.strip()
        else:
          # check if subject has substance
          subs_match = substance_param in s.strip().lower()
          if subs_match:
            new_subject = s.strip()
  if 'new_subject' in locals():
    return new_subject
  else:
    print("There is no hasPoint or isPointOf relation.")
    return None

# Add new triple using subject found in the existing data-model triples

if __name__ == "__main__":

  dataset = "AHU_prin_winter_2023_stanscaler_RULES"
  # Specify the filename of JSON file having all class predictions (substance, comp, subsys)
  read_filename = os.path.join(dataset+'_pointwise_preds_dict.json')

  # Open the file in read mode with appropriate encoding (assuming UTF-8)
  with open(read_filename, "r", encoding="utf-8") as f:
    # Use json.load to parse the JSON data from the file
    classification_dict = json.load(f)
  # classification_dict = dict()
  # classification_dict['fAHUTempODAADS'] = {'substance':'temp', 
  #                                          'component':None,
  #                                          'subsystem':'ODA'}
  # classification_dict['fAHUPHValveActADS'] = {'substance':'valve', 
  #                                             'component':'heater',
  #                                             'subsystem':'ODA'}

  for key in classification_dict:
    component_param = classification_dict[key]['component']
    subsystem_param = classification_dict[key]['subsystem']
    substance_param = classification_dict[key]['substance']
    print(f"\nFinding metadata family for: {key}")
    if component_param:
      new_subject = find_iterative_matches(component_param.lower(), substance_param.lower())
      print(new_subject)
    elif subsystem_param:
      new_subject = find_iterative_matches(subsystem_param.lower(), substance_param.lower())
      print(new_subject)
    
    literal_subject = new_subject[new_subject.find("#") + 1:]
    
    bldg_prefix = Namespace('http://example.com/mybuilding#')
    graph.bind('bldg',bldg_prefix)
    ref_prefix = Namespace("https://brickschema.org/schema/Brick/ref#")
    graph.bind('ref', ref_prefix)
    # New information to add (modify as needed)
    ref_predicate = ref_prefix.hasExternalReference
    object_term = literal_subject + "_ref"
    ref_object = bldg_prefix[object_term]
    ref_subject = bldg_prefix[literal_subject]

    graph.add((ref_subject, ref_predicate, ref_object))
    print(f"Added new triple: ({graph.qname(ref_subject)}, {graph.qname(ref_predicate)}, {graph.qname(ref_object)})")

    timeseries_predicate = ref_prefix.hasTimeseriesId
    timeseries_object = bldg_prefix[key]
    graph.add((ref_object, timeseries_predicate, timeseries_object))
    print(f"Added new triple: ({graph.qname(ref_object)}, {graph.qname(timeseries_predicate)}, {graph.qname(timeseries_object)})")

# Iterate through the triples (subject, predicate, object) in the graph
# for subject, predicate, object in graph:
#   # Check if the subject matches your search parameter
  # if predicate.strip() == predicate_search_param and object.strip() == subsystem_param:
  #   # You've found a triple where the subject matches your parameter
  #   print(f"Subject: {subject}")
  #   print(f"Predicate: {predicate}")
  #   print(f"Object: {object}")
  #   print("---")  # Separator for readability

  #   result_subject = subject
  #   # Find all triples with the matching subject
  #   all_triples = find_triples_by_subject(subject)
  #   print(f"All triples with subject '{subject}':")
  #   for s, p, o in all_triples:
  #     print(f"\t- ({s}, {p}, {o})")
  #   # Here, you can further process the data structure based on your needs