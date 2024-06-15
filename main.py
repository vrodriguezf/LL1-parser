import pandas as pd
from tabulate import tabulate
from cfg import *
import sys

def extract_grammar(string, rule_index, starting_non_terminal, non_terminals, terminals, left_side, right_side, rules, RHS, priority):
    split = string.split()
    rhs_i = []
    last_l = 2
    non_terminal = split[0]
    if non_terminal.islower():
        raise ValueError("Non-terminals should be uppercase letters.")
    if rule_index == 0:
        starting_non_terminal = non_terminal
    non_terminals.add(non_terminal)
    priority[non_terminal] = rule_index
    rules_i = [non_terminal, "->"]
    for i in range(2, len(split)):
        if split[i].isupper():
            rhs_i.insert(0, split[i])
            rules_i.append(split[i])
            non_terminals.add(split[i])
            if split[i] in right_side:
                right_side[split[i]].add((rule_index, i - last_l))
            else:
                right_side[split[i]] = {(rule_index, i - last_l)}
        elif split[i].islower():
            if split[i] == "micro":
                raise ValueError("Do not use 'micro' as a terminal.")
            if i - last_l != 0:
                rhs_i.insert(0, "micro")
                rhs_i.insert(0, split[i])
            else:
                rhs_i.insert(0, split[i])
            terminals.add(split[i])
            rules_i.append(split[i])
        elif split[i] == "|":
            if i - last_l == 0:
                raise ValueError("Invalid use of '|' at the start of a rule.")
            else:
                RHS.append(rhs_i)
                rules.append(rules_i)
                if non_terminal in left_side:
                    left_side[non_terminal].add(rule_index)
                else:
                    left_side[non_terminal] = {rule_index}
                rules_i = [non_terminal, "->"]
                rhs_i = []
                rule_index += 1
                last_l = i + 1
        elif split[i] in ["->", "$"]:
            raise ValueError(f"Do not use '{split[i]}' as a terminal.")
        else:
            if i - last_l != 0:
                rhs_i.insert(0, "micro")
                rhs_i.insert(0, split[i])
            else:
                if split[i] != '""':
                    rhs_i.insert(0, split[i])
            if split[i] != '""':
                terminals.add(split[i])
            rules_i.append(split[i])
    RHS.append(rhs_i)
    rules.append(rules_i)
    if non_terminal in left_side:
        left_side[non_terminal].add(rule_index)
    else:
        left_side[non_terminal] = {rule_index}
    rule_index += 1
    return rule_index, starting_non_terminal


def update_first(non_terminal, left_side, first, terminals, rules):
    update = False
    before = first[non_terminal].copy()
    for l_pos in left_side[non_terminal]:
        if rules[l_pos][2] in terminals or rules[l_pos][2] == '""':
            first[non_terminal].add(rules[l_pos][2])
        else:
            for i in range(2, len(rules[l_pos])):
                if rules[l_pos][i] in terminals:
                    first[non_terminal].add(rules[l_pos][i])
                    break
                else:
                    u_set = first[rules[l_pos][i]].copy()
                    if '""' in u_set:
                        if i != len(rules[l_pos]) - 1:
                            u_set.remove('""')
                        first[non_terminal].update(u_set)
                    else:
                        first[non_terminal].update(u_set)
                        break
    if before != first[non_terminal]:
        update = True
    return update



def update_follow(non_terminal, follow, right_side, rules, first, terminals):
    update = False
    before = follow[non_terminal].copy()
    if non_terminal not in right_side:
        return update
    for r_pos in right_side[non_terminal]:
        rule = rules[r_pos[0]]
        if r_pos[1] + 2 == len(rule) - 1:
            follow[non_terminal].update(follow[rule[0]])
        else:
            for i in range(r_pos[1] + 3, len(rule)):
                if rule[i] in terminals:
                    follow[non_terminal].add(rule[i])
                    break
                u_set = first[rule[i]].copy()
                if '""' in u_set:
                    u_set.remove('""')
                    follow[non_terminal].update(u_set)
                    if i == len(rule) - 1:
                        follow[non_terminal].update(follow[rule[0]])
                else:
                    follow[non_terminal].update(u_set)
                    break
    if before != follow[non_terminal]:
        update = True
    return update


def is_useless(non_terminal, left_side, right_side, starting_non_terminal, RHS):
    if non_terminal not in left_side:
        return True
    if non_terminal not in right_side and non_terminal != starting_non_terminal:
        return True
    for i in left_side[non_terminal]:
        if non_terminal not in RHS[i]:
            return False
    return True


def predict(index, rules, first, follow, terminals):
    rule = rules[index]
    index_predict_set = set()
    for i in range(2, len(rule)):
        if rule[i] in terminals:
            index_predict_set.add(rule[i])
            break
        if rule[i] == '""':
            index_predict_set.update(follow[rule[0]])
            break
        u_set = first[rule[i]].copy()
        if '""' in first[rule[i]]:
            u_set.remove('""')
            index_predict_set.update(u_set)
            if i == len(rule) - 1:
                index_predict_set.update(follow[rule[0]])
        else:
            index_predict_set.update(u_set)
            break
    return index_predict_set


def construct_parse_table(non_terminals, terminals, left_side, rules, predict_set):
    is_ll1 = True
    parse_table = {nt: {} for nt in non_terminals}

    for t in terminals:
        for nt in non_terminals:
            for i in left_side[nt]:
                if t in predict_set[i]:
                    if t in parse_table[nt]:
                        is_ll1 = False
                    else:
                        parse_table[nt][t] = []
                    parse_table[nt][t].append(i)
            if t not in parse_table[nt]:
                parse_table[nt][t] = [-1]
    return parse_table, is_ll1


def main(grammar_file, positive_tests=None, negative_tests=None):
    left_side = {}
    right_side = {}
    first = {}
    follow = {}
    RHS = []
    rules = []
    predict_set = []
    non_terminals = set()
    starting_non_terminal = None
    terminals = set()
    parse_table = {}
    priority = {}
    
    with open(grammar_file, 'r') as file:
        lines = file.readlines()
    
    rule_index = 0
    for line in lines:
        line = line.strip()
        if line:
            rule_index, starting_non_terminal = extract_grammar(
                line, rule_index, starting_non_terminal, non_terminals, terminals, left_side, right_side, rules, RHS, priority
            )
    
    for T in non_terminals:
        first[T] = set()
        follow[T] = set()
    
    follow[starting_non_terminal].add("$")
    
    update = True
    while update:
        update = False
        for T in non_terminals:
            update |= update_first(T, left_side, first, terminals, rules)
    
    update = True
    while update:
        update = False
        for T in non_terminals:
            update |= update_follow(T, follow, right_side, rules, first, terminals)
    
    for j in range(len(rules)):
        predict_set.append(predict(j, rules, first, follow, terminals))
    
    terminals.add("$")
    parse_table, ll1 = construct_parse_table(non_terminals, terminals, left_side, rules, predict_set)
    terminals.remove("$")
    terminals_list = list(terminals)
    terminals_list.append("$")
    non_terminals_list = sorted(non_terminals, key=lambda x: priority[x])
    
    f_f_n_for_show = {
        "nullable": [],
        "first": [],
        "follow": []
    }
    
    for ntr in non_terminals_list:
        if '""' in first[ntr]:
            f_f_n_for_show["nullable"].append("Y")
        else:
            f_f_n_for_show["nullable"].append("N")
        f_f_n_for_show["first"].append(' '.join(first[ntr]))
        f_f_n_for_show["follow"].append(' '.join(follow[ntr]))
    
    f_f_n_for_show["nullable"] = pd.Series(f_f_n_for_show["nullable"], index=non_terminals_list)
    f_f_n_for_show["first"] = pd.Series(f_f_n_for_show["first"], index=non_terminals_list)
    f_f_n_for_show["follow"] = pd.Series(f_f_n_for_show["follow"], index=non_terminals_list)
    
    f_f_n_df = pd.DataFrame(f_f_n_for_show)
    
    rules_for_show = {"Production Rules": pd.Series(' '.join(e) for e in rules)}
    RHS_for_show = {"RHS": pd.Series(' '.join(e) for e in RHS)}
    
    rules_df = pd.DataFrame(rules_for_show)
    RHS_df = pd.DataFrame(RHS_for_show)
    
    parse_table_for_show = {tr: [] for tr in terminals_list}
    
    for tr in terminals_list:
        for ntr in non_terminals_list:
            parse_table_for_show[tr].append(' '.join(str(e) for e in parse_table[ntr][tr]))
    
    parse_table_df = pd.DataFrame(parse_table_for_show, index=non_terminals_list)
    
    print(f"{tabulate(rules_df, headers='keys', tablefmt='fancy_grid')}\n")
    print(f"nullable/first/follow table:\n{tabulate(f_f_n_df, headers='keys', tablefmt='fancy_grid')}\n")

    # Construct a dictionary to store the predict sets for each rule
    predict_sets_for_show = {"Rule": [], "Predict Set": []}
    for idx, predict_set_rule in enumerate(predict_set):
        predict_sets_for_show["Rule"].append(' '.join(rules[idx]))
        predict_sets_for_show["Predict Set"].append(' '.join(predict_set_rule))

    # Convert to DataFrame
    predict_sets_df = pd.DataFrame(predict_sets_for_show)

    # Print the predict sets DataFrame
    print(f"Predict Sets for each rule:\n{tabulate(predict_sets_df, headers='keys', tablefmt='fancy_grid')}\n")

    print(f"And this is the parse table:\n{tabulate(parse_table_df, headers='keys', tablefmt='fancy_grid')}\n")
    if not ll1:
        print("GRAMMAR IS NOT LL(1)\n")
    else:
        print("GRAMMAR IS LL(1)\n")

    # Use the cfg module and convert the grammar to CFG
    # CFG
    # Parameters
    #     variables (optional): grammar's variables set.
    #     terminals: grammar's terminals set
    #     rules:  grammar's rules
    #     start_variable (optional, defaults to 'S'): grammar's start variable
    #     null_character (optional, defaults to 'λ'): grammar's null character
    # Example:
    # g = CFG(terminals={'a', 'b', 'c', 'λ'},
    #     rules={'S': ['aSa', 'bSb', 'cSc', 'λ']}
    #     )

    # Turn the list of rules into a dictionary for cfg
    rules_dict = {}
    for rule in rules:
        # Join the elements of rule[2:] into a single string
        rhs = ''.join(rule[2:])
        # If the string is empty, replace it with epsilon
        if rhs == '""':
            rhs = "ε"
        if rule[0] in rules_dict:
            rules_dict[rule[0]].append(rhs)
        else:
            rules_dict[rule[0]] = [rhs]
    
    # Create the CFG object
    print("Converting the grammar to pyCFG format...")
    g = CFG(terminals=terminals.union('ε'), 
            variables=non_terminals,
            rules=rules_dict, 
            start_variable=starting_non_terminal, 
            null_character='ε')
    
    print(g)

    if positive_tests:
            with open(positive_tests, 'r') as file:
                p_strings = [line.strip() for line in file if line.strip()]        
            p_strings_for_show = {"String": [], "Is Accepted": []}
            for p_string in p_strings:
                p_strings_for_show["String"].append(p_string)
                p_strings_for_show["Is Accepted"].append(g.cyk(p_string))
            p_strings_df = pd.DataFrame(p_strings_for_show)
            print(f"\nPositive Tests:\n{tabulate(p_strings_df, headers='keys', tablefmt='fancy_grid')}\n")
    
    if negative_tests:
        with open(negative_tests, 'r') as file:
            n_strings = [line.strip() for line in file if line.strip()]
        n_strings_for_show = {"String": [], "Is Accepted": []}
        for n_string in n_strings:
            n_strings_for_show["String"].append(n_string)
            n_strings_for_show["Is Accepted"].append(g.cyk(n_string))
        n_strings_df = pd.DataFrame(n_strings_for_show)
        print(f"Negative Tests:\n{tabulate(n_strings_df, headers='keys', tablefmt='fancy_grid')}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python ll1_parser.py <grammar_file> [positive_tests] [negative_tests]")
        sys.exit(1)
    grammar_file = sys.argv[1]
    positive_tests = sys.argv[2] if len(sys.argv) >= 3 else None
    negative_tests = sys.argv[3] if len(sys.argv) == 4 else None
    main(grammar_file, positive_tests, negative_tests)
