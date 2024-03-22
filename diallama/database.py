import json
import random
import copy
import os
from typing import Text, Dict

from fuzzywuzzy import fuzz
from dateutil import parser
from datetime import datetime
import re


class MultiWOZDatabase:
    """ MultiWOZ database implementation. """

    IGNORE_VALUES = {
        'hospital' : ['id'],
        'police' : ['id'],
        'attraction' : ['location', 'openhours'],
        'hotel' : ['location', 'price'],
        'restaurant' : ['location', 'introduction']
    }

    FUZZY_KEYS = {
        'hospital' : {'department'},
        'hotel' : {'name', 'area'},
        'attraction' : {'name'},
        'restaurant' : {'name', 'food', 'area'},
        'bus' : {'departure', 'destination'},
        'train' : {'departure', 'destination'},
        'police' : {'name'}
    }

    DOMAINS = [
        'restaurant',
        'hotel',
        'attraction',
        'train',
        'taxi',
        'police',
        'hospital'
    ]

    def __init__(self):
        self.data, self.data_keys = self._load_data()

    def _load_data(self):
        database_data = {}
        database_keys = {}

        for domain in self.DOMAINS:
            with open(os.path.join(os.path.dirname(__file__), "database", f"{domain}_db.json"), "r") as f:
                for l in f:
                    if not l.startswith('##') and l.strip() != "":
                        f.seek(0)
                        break
                database_data[domain] = json.load(f)

            if domain in self.IGNORE_VALUES:
                for i in database_data[domain]:
                    for ignore in self.IGNORE_VALUES[domain]:
                        if ignore in i:
                            i.pop(ignore)

            database_keys[domain] = set()
            if domain == 'taxi':
                database_data[domain] =  {k.lower(): v for k, v in database_data[domain].items()}
                database_keys[domain].update([k.lower() for k in database_data[domain].keys()])
            else:
                for i, database_item in enumerate(database_data[domain]):
                    database_data[domain][i] =  {k.lower(): v for k, v in database_item.items()}
                    database_keys[domain].update([k.lower() for k in database_item.keys()])

        return database_data, database_keys

    def time_str_to_minutes(self, time_string) -> Text:
        """ Converts time to the only format supported by database, i.e. HH:MM in 24h format
            For example: "noon" -> 12:00
        """

        number_mapping = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
            'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90
        }

        word_mapping = {
            "noon": "12:00",
            "lunch": "10:00",
            "morning": "6:00",
            "midnight": "00:00",
            "night": "21:00",
            "evening": "18:00",
            "afternoon": "15:00",
            "forenoon": "9:00"
        }

        if not bool(re.search(r'\d', time_string)):
            words = time_string.lower().split()

            hours, minutes = 0, 0
            result = ""

            for word in words:
                if word in number_mapping:
                    if hours == 0:
                        hours += number_mapping[word]
                        result += str(hours)
                    else:
                        minutes += number_mapping[word]

                        if word == words[-1]:
                            result += f":{str(minutes)} "

                elif word in word_mapping:
                    result += word_mapping[word]

                else:
                    if minutes == 0:
                        result += ":00 " if hours != 0 else ""
                    else:
                        result += f":{str(minutes)} "

                    result += word

            time_string = result

        default_time = "00:00"

        if bool(re.search(r'\d', time_string)):
            time_string = time_string.replace(".", ":")
            if (" " not in time_string) and (re.search(r'[a-zA-Z]+', time_string)) and (re.search(r'[0-9]+', time_string)):
                digits = ''.join(filter(str.isdigit, time_string))
                time_string = ':'.join(digits[i:i+2] for i in range(0, len(digits), 2))

            try:
                parsed_time = parser.parse(time_string, fuzzy=True)
                time_part = parsed_time.time()
                converted_time_string = time_part.strftime("%H:%M")
            except parser.ParserError:
                converted_time_string = default_time
                time_string += " after"

        else:
            converted_time_string = default_time
            time_string += " after"

        if "after" in time_string:
            converted_time_string = f"> {converted_time_string}"
        elif "before" in time_string:
            converted_time_string = f"< {converted_time_string}"
        else:
            converted_time_string = f"= {converted_time_string}"

        return converted_time_string


    def query(self,
              domain: Text,
              constraints: Dict[Text, Text],
              fuzzy_ratio: int=90):
        """
        Returns the list of entities (dictionaries) for a given domain based on the annotation of the belief state.

        Arguments:
            domain:      Name of the queried domain.
            constraints: Hard constraints to the query results.
        """

        if domain == 'taxi':
            c, t, p = None, None, None

            c = constraints.get('color', [random.choice(self.data[domain]['taxi_colors'])])[0]
            t = constraints.get('type', [random.choice(self.data[domain]['taxi_types'])])[0]
            p = constraints.get('phone', [''.join([str(random.randint(1, 9)) for _ in range(11)])])[0]

            return [{'color': c, 'type' : t, 'phone' : p}]

        elif domain == 'hospital':

            hospital = {
                'hospital phone': '01223245151',
                'address': 'Hills Rd, Cambridge',
                'postcode': 'CB20QQ',
                'name': 'Addenbrookes'
            }

            departments = [x.strip().lower() for x in constraints.get('department', [])]
            phones = [x.strip().lower() for x in constraints.get('phone', [])]

            if len(departments) == 0 and len(phones) == 0:
                return [dict(hospital)]
            else:
                results = []
                for i in self.data[domain]:
                    if 'department' in self.FUZZY_KEYS[domain]:
                        f = (lambda x: fuzz.partial_ratio(i['department'].lower(), x) > fuzzy_ratio)
                    else:
                        f = (lambda x: i['department'].lower() == x)

                    if any(f(x) for x in departments) and \
                       (len(phones) == 0 or any(i['phone'] == p.strip() for p in phones)):
                        results.append(dict(i))
                        results[-1].update(hospital)

                return results

        else:
            # Hotel database keys:      address, area, name, phone, postcode, pricerange, type, internet, parking, stars, takesbookings (other are ignored)
            # Attraction database keys: address, area, name, phone, postcode, pricerange, type, entrance fee (other are ignored)
            # Restaurant database keys: address, area, name, phone, postcode, pricerange, type, food

            # Train database contains keys: arriveby, departure, day, leaveat, destination, trainid, price, duration
            # The keys arriveby, leaveat expect a time format such as 8:45 for 8:45 am

            results = []
            query = {}

            if domain == 'attraction' and 'entrancefee' in constraints:
                constraints['entrance fee'] = constraints.pop(['entrancefee'])

            if domain not in self.data_keys.keys():
                return {}

            for key in self.data_keys[domain]:
                query[key] = constraints.get(key, [])
                if len(query[key]) > 0 and key in ['arriveby', 'leaveat']:
                    if isinstance(query[key][0], str):
                        query[key] = [query[key]]
                    query[key] = [self.time_str_to_minutes(x) for x in query[key]]
                    query[key] = list(set(query[key]))

            for i, item in enumerate(self.data[domain]):
                for k, v in query.items():
                    if len(v) == 0 or item[k] == '?':
                        continue

                    if k == 'arriveby':
                        t = f"00{item[k][2:]}" if item[k].startswith("24") else item[k]

                        q_time = datetime.strptime(v[0][2:], "%H:%M").time()
                        item_time = datetime.strptime(t, "%H:%M").time()

                        if item_time > q_time and v[0][:1] != ">":
                            break

                        # accept item[k] if it is earlier or equal to time in the query
                        pass
                    elif k == 'leaveat':
                        t = f"00{item[k][2:]}" if item[k].startswith("24") else item[k]

                        q_time = datetime.strptime(v[0][2:], "%H:%M").time()
                        item_time = datetime.strptime(t, "%H:%M").time()

                        if item_time < q_time and v[0][:1] != "<":
                            break

                        # accept item[k] if it is later or equal to time in the query
                        pass
                    else:

                        if k in self.FUZZY_KEYS[domain]:
                            match = fuzz.partial_ratio(item[k], v)
                            if match < 50:
                                break
                        else:
                            if item[k].lower() != v.lower():
                                break

                        #  accept item[k] if it matches to the values in query
                        pass

                else: # This gets executed iff the above loop is not terminated
                    result = copy.deepcopy(item)
                    if domain in ['train', 'hotel', 'restaurant']:
                        ref = constraints.get('ref', [])
                        result['ref'] = '{0:08d}'.format(i) if len(ref) == 0 else ref

                    results.append(result)

            if domain == 'attraction':
               for result in results:
                   result['entrancefee'] = result.pop('entrance fee')

            return results
