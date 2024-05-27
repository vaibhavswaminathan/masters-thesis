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

def find_reference_in_graph(datapoint):
  triples = False
  ext_ref_param = "https://brickschema.org/schema/Brick/ref#hasExternalReference"
  results = graph.query(
        """
        SELECT *
        WHERE {
        %s %s ?object .
        }
        """ % (graph.qname(datapoint), graph.qname(ext_ref_param)))

  if results:
    print("CLASH")
    triples = True
    for row in results:
        print(row)  

  return triples

def find_iterative_matches(comp_param, substance_param):
  """
  Search all triples in graph firstly for:
  1. Component(comp_param) / substance(substance_param) and secondly,
  2. brick:hasPoint or brick:isPointOf relations
  Return the subject to use for adding new triple
  """
  
  print(f"Pred. Comp: {comp_param} -------- Pred. Substance: {substance_param}")

  if substance_param == "temperature":
    substance_param = "temp"

  key_triples = find_keyword_in_graph(comp_param)
  # print("Key Triples found using comp_param: \n")
  # for s,p,o in key_triples:
  #   print(f"\t- ({s}, {p}, {o})")
  if not key_triples:
    print(f"No triples found having: {comp_param}")
    return None
  for s, p, o in key_triples:
      # print(f"\t- ({s}, {p}, {o})")
      # check for brick:hasPoint or brick:isPointOf relations
      
      if p.strip() == has_point_param:
        res_match = has_point_param
      elif p.strip() == is_pointof_param:
        res_match = is_pointof_param
      else:
        res_match  = None

      if res_match != None:
        # print(res_match)
        if res_match.strip() == has_point_param:
          # check if object has substance
          subs_match = substance_param in o.strip().lower()
          if subs_match:
            already_added = find_reference_in_graph(o.strip()) # check if o.strip() already has external reference
            if not already_added:
              new_subject = o.strip()
              print(f"New subject: {new_subject}")
            else:
              print("Ref already exists")
        else:
          # check if subject has substance
          subs_match = substance_param in s.strip().lower()
          if subs_match:
            already_added = find_reference_in_graph(s.strip()) # check if s.strip() already has external reference
            if not already_added:
              new_subject = s.strip()
              print(f"New subject: {new_subject}")
            else:
              print("Ref already exists")
  if 'new_subject' in locals(): # check if 'new_subject' variable has been created
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

  # Data format in dictionary:
  # classification_dict['fAHUPHValveActADS'] = {'substance':'valve', 
  #                                             'component':'heater',
  #                                             'subsystem':'ODA'}

  for key in classification_dict:
    component_param = classification_dict[key]['component']
    subsystem_param = classification_dict[key]['subsystem']
    substance_param = classification_dict[key]['substance']
    print(f"\nFinding metadata family for: {key}")
    print(f"Pred. Substance: {substance_param}")
    print(f"Pred. Component: {component_param}")
    print(f"Pred. Subsystem: {subsystem_param}")
    existing_param = None
    new_subject = ""
    if component_param:
      existing_param = component_param
      new_subject = find_iterative_matches(component_param.lower(), substance_param.lower())
      print(new_subject)
    elif subsystem_param:
      existing_param = subsystem_param
      new_subject = find_iterative_matches(subsystem_param.lower(), substance_param.lower())
      print(new_subject)
    else:
      print("Only Substance available")
      new_subject = substance_param + "_point"
    
    # Define namespace prefixes
    bldg_prefix = Namespace('http://example.com/mybuilding#')
    graph.bind('bldg',bldg_prefix)
    brick_prefix = Namespace('https://brickschema.org/schema/Brick#')
    graph.bind('brick', brick_prefix)
    ref_prefix = Namespace("https://brickschema.org/schema/Brick/ref#")
    graph.bind('ref', ref_prefix)

    ref_predicate = ref_prefix.hasExternalReference
    timeseries_predicate = ref_prefix.hasTimeseriesId
    timeseries_object = bldg_prefix[key]
    
    if new_subject is None:
      point_name = "someName"
      point_ref = "someName" + "_ref"
      graph.add((bldg_prefix[existing_param], brick_prefix.hasPoint, bldg_prefix[point_name])) # existing_param hasPoint someName
      graph.add((bldg_prefix[point_name], ref_predicate, bldg_prefix[point_ref])) # someName hasExternalRef someRef
      graph.add((bldg_prefix[point_ref], timeseries_predicate, timeseries_object))
       # someRef hasTimeseriesId datapoint_name

      print(f"Added new triple: ({graph.qname(bldg_prefix[existing_param])}, {graph.qname(brick_prefix.hasPoint)}, {graph.qname(bldg_prefix[point_name])})")
      print(f"Added new triple: ({graph.qname(bldg_prefix[point_name])}, {graph.qname(ref_predicate)}, {graph.qname(bldg_prefix[point_ref])})")
      print(f"Added new triple: ({graph.qname(bldg_prefix[point_ref])}, {graph.qname(timeseries_predicate)}, {graph.qname(timeseries_object)})")
    else:
      literal_subject = new_subject[new_subject.find("#") + 1:]
      
      # New information to add (modify as needed)
      object_term = literal_subject + "_ref"
      ref_object = bldg_prefix[object_term]
      ref_subject = bldg_prefix[literal_subject]

      graph.add((ref_subject, ref_predicate, ref_object))
      print(f"Added new triple: ({graph.qname(ref_subject)}, {graph.qname(ref_predicate)}, {graph.qname(ref_object)})")

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