"""
Protótipo híbrido: Algoritmo Genético + Programação Linear + RSP (Resource Scheduling Problem)
Gera horários para turmas 5A..9B e 1M..3M (25 aulas/semana).
Salva saída em 'hybrid_schedule.xlsx'.

Notas:
 - Este é um protótipo para demonstrar a combinação de técnicas.
 - Para usos em produção é recomendado migrar para OR-Tools / DEAP e tratar mais constraints.
"""

import random
import math
from collections import Counter, defaultdict
import pandas as pd
import copy
import pulp

# --------------------------
# Configurações básicas
# --------------------------
DAYS = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
PERIODS_PER_DAY = 5
SLOTS = [(d, p+1) for d in DAYS for p in range(PERIODS_PER_DAY)]  # 25 slots linear order
SERIES_FUND = ["6A","6B","7A","7B","8A","8B","9A","9B"]
SERIES_MED = ["1M","2M","3M"]
ALL_TURMAS = SERIES_FUND + SERIES_MED
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Curricula (garantindo 25 aulas)
CURRICULUM_FUND = {
    "Português": 5, "Matemática": 5, "História": 3, "Geografia": 3,
    "Ciências": 3, "Inglês": 2, "Educação Física": 2, "Arte": 2
}
CURRICULUM_MED = {
    "Português": 5, "Matemática": 5, "Física": 2, "Química": 2, "Biologia": 2,
    "História": 2, "Geografia": 2, "Sociologia": 2, "Inglês": 2, "Filosofia": 1
}

def build_curriculum(turma):
    if turma in SERIES_FUND:
        base = dict(CURRICULUM_FUND)
        total = sum(base.values()); diff = 25 - total
        base["Atividades Complementares"] = base.get("Atividades Complementares",0) + diff
        return base
    else:
        base = dict(CURRICULUM_MED)
        total = sum(base.values()); diff = 25 - total
        if diff>0:
            base["Atividades Complementares"] = base.get("Atividades Complementares",0) + diff
        return base

# --------------------------
# Entrada: professores (exemplo)
# --------------------------
# Estrutura: professors = [{"name":str,"subjects":[...],"availability": {(day)->list(periods)} }]
def example_professors():
    # Agora cada professor tem:
    #  - "classes": lista das turmas que ele leciona
    #  - "availability": dict dia->lista de períodos disponíveis (1..PERIODS_PER_DAY)
    full_avail = {d: list(range(1, PERIODS_PER_DAY+1)) for d in DAYS}
    profs = [
        {"name":"Priscila","subjects":["Português"], "classes":["6A","7A"], "availability": full_avail},
        {"name":"Joao","subjects":["Português"], "classes":["6B","7B"], "availability": full_avail},
        {"name":"Edna","subjects":["Matemática"], "classes":["6B","7B"], "availability": full_avail},
        {"name":"Diogo","subjects":["Física"], "classes":["1M"], "availability": full_avail},
        {"name":"Murilo","subjects":["Química"], "classes":["2M"], "availability": full_avail},
        {"name":"Valdemar","subjects":["Biologia"], "classes":["3M"], "availability": full_avail},
        {"name":"Marleide","subjects":["Ciências"], "classes":["6A","6B"], "availability": full_avail},
        {"name":"Karen","subjects":["História"], "classes":["8A","8B"], "availability": full_avail},
        {"name":"Rafael","subjects":["Geografia"], "classes":["8A","9A"], "availability": full_avail},
        {"name":"Virginia","subjects":["Inglês"], "classes":["9B","1M"], "availability": full_avail},
        {"name":"Ana","subjects":["Educação Física"], "classes":["6A","7A","8A"], "availability": full_avail},
        {"name":"Cassio","subjects":["Arte"], "classes":["6B","7B","8B"], "availability": full_avail},
        {"name":"Bob","subjects":["Filosofia","Sociologia"], "classes":["2M","3M"], "availability": full_avail},
    ]
    return profs

# --------------------------
# Representação de solução (cromossomo)
# --------------------------
# Para cada turma teremos uma lista de 25 disciplinas (em ordem SLOTS)
# Cromossomo = dict { turma: [disciplina_index_0..24] } (strings)

def build_initial_individual(turma):
    # Gera uma sequência de disciplinas respeitando carga horária (apenas multiconjunto)
    curriculum = build_curriculum(turma)
    multiset = []
    for subj, h in curriculum.items():
        multiset += [subj]*h
    random.shuffle(multiset)
    return multiset  # length 25

# populaçao inicial
def initial_population(pop_size=40):
    pop = []
    for _ in range(pop_size):
        indiv = {}
        for t in ALL_TURMAS:
            indiv[t] = build_initial_individual(t)
        pop.append(indiv)
    return pop

# --------------------------
# RSP: Validação de recursos
# --------------------------
# Verifica conflitos de professores e salas.
# Para simplificação: cada disciplina mapeia a um conjunto de professores possíveis (input).
# Também modelamos um número fixo de salas gerais por turma (cada turma tem sua sala, logo não há conflito de sala entre turmas).
# Mas incluímos verificação de conflitos de professor (um professor não pode estar em 2 turmas no mesmo slot).

def evaluate_resource_violations(indiv, profs_by_subject, rooms_per_turma=None):
    # profs_by_subject: {subject: [prof_dicts]}
    violations = 0
    slot_assignments = defaultdict(list)
    for t in ALL_TURMAS:
        schedule = indiv[t]
        for idx, subj in enumerate(schedule):
            day, period = SLOTS[idx]
            slot_assignments[(day,period)].append((t, subj))
    chosen_teacher = {}
    teacher_occupied = defaultdict(set)  # (day,period) -> set(teacher)
    for slot, entries in slot_assignments.items():
        day, period = slot
        random.shuffle(entries)
        for turma, subj in entries:
            candidates = profs_by_subject.get(subj, [])
            selected = None
            # 1) prefer candidate who teaches this turma and is available this slot and not occupied
            for cand in candidates:
                if turma in cand.get("classes",[]) and day in cand.get("availability",{}) and period in cand["availability"][day] and cand["name"] not in teacher_occupied[slot]:
                    selected = cand["name"]
                    break
            # 2) otherwise prefer any candidate available this slot and not occupied
            if selected is None:
                for cand in candidates:
                    if day in cand.get("availability",{}) and period in cand["availability"][day] and cand["name"] not in teacher_occupied[slot]:
                        selected = cand["name"]
                        break
            # 3) otherwise pick a candidate who teaches the turma (may be unavailable)
            if selected is None:
                for cand in candidates:
                    if turma in cand.get("classes",[]):
                        selected = cand["name"]
                        break
            # 4) otherwise fallback to first candidate (may be wrong subject mapping)
            if selected is None and candidates:
                selected = candidates[0]["name"]
            if selected is None:
                # no teacher for that subject at all -> hard violation
                violations += 5
                chosen_teacher[(turma,slot)] = None
            else:
                # check details of candidate to possibly add soft violations
                cand_obj = next((c for c in candidates if c["name"]==selected), None)
                if cand_obj:
                    # if teacher not available this slot -> count small violation
                    if not (day in cand_obj.get("availability",{}) and period in cand_obj["availability"].get(day,[])):
                        violations += 2
                    # if teacher doesn't actually teach that turma -> small penalty
                    if turma not in cand_obj.get("classes",[]):
                        violations += 1
                # if selected already occupied at slot -> conflict
                if selected in teacher_occupied[slot]:
                    violations += 3
                teacher_occupied[slot].add(selected)
                chosen_teacher[(turma,slot)] = selected
    return violations, chosen_teacher

# --------------------------
# Fitness (GA)
# --------------------------
# Objetivos:
#  - penalizar repetições >2 no mesmo dia
#  - penalizar matérias não espalhadas (prefere spread)
#  - penalizar violações de RSP (professor conflitante / sem professor)
#  - small bonus: balancear assuntos por dia
def fitness(indiv, profs_by_subject):
    score = 0.0
    # 1) penalizar repetições >2 consecutivas no mesmo dia por turma
    for t in ALL_TURMAS:
        sched = indiv[t]
        for d_i, day in enumerate(DAYS):
            day_slice = sched[d_i*PERIODS_PER_DAY:(d_i+1)*PERIODS_PER_DAY]
            # count runs
            run_subject = None; run_len = 0
            for subj in day_slice:
                if subj == run_subject:
                    run_len += 1
                else:
                    if run_len > 2:
                        score -= (run_len - 2) * 1.0
                    run_subject = subj; run_len = 1
            if run_len > 2:
                score -= (run_len - 2) * 1.0
    # 2) penalizar distribuição ruim: preferir que a same subject be spread across days
    for t in ALL_TURMAS:
        counts_per_subj = Counter(indiv[t])
        # prefer counts matched to curriculum (we assume they match); reward uniformity across days:
        for subj in counts_per_subj:
            # compute how many days that subject appears in this turma
            days_present = 0
            for d_i in range(len(DAYS)):
                slice_subjs = set(indiv[t][d_i*PERIODS_PER_DAY:(d_i+1)*PERIODS_PER_DAY])
                if subj in slice_subjs: days_present += 1
            # encourage subject to appear in multiple days (unless it has very few hours)
            if counts_per_subj[subj] >= 3:
                score += 0.1 * days_present
            else:
                score += 0.02 * days_present
    # 3) RSP violations
    violations, _ = evaluate_resource_violations(indiv, profs_by_subject)
    score -= violations * 1.0
    # 4) small randomness to diversify
    score += random.random()*0.001
    return score

# --------------------------
# GA operators (simple)
# --------------------------
def tournament_selection(pop, scores, k=3):
    selected = random.sample(list(range(len(pop))), k)
    selected.sort(key=lambda i: scores[i], reverse=True)
    return copy.deepcopy(pop[selected[0]])

def crossover(parent1, parent2):
    # one-point crossover per turma: choose random cut and swap tails
    child1 = {}
    child2 = {}
    for t in ALL_TURMAS:
        cut = random.randint(1, len(parent1[t])-1)
        a = parent1[t][:cut] + parent2[t][cut:]
        b = parent2[t][:cut] + parent1[t][cut:]
        # repair to ensure counts match curriculum: do simple multiset repair
        a = repair_to_curriculum(a, t)
        b = repair_to_curriculum(b, t)
        child1[t] = a; child2[t] = b
    return child1, child2

def mutate(indiv, mut_rate=0.02):
    # swap two positions within a random turma
    for t in ALL_TURMAS:
        if random.random() < mut_rate:
            i,j = random.sample(range(25),2)
            indiv[t][i], indiv[t][j] = indiv[t][j], indiv[t][i]
    return indiv

def repair_to_curriculum(arr, turma):
    # ensure arr contains exact multiset as curriculum
    desired = build_curriculum(turma)
    desired_list = []
    for subj,h in desired.items():
        desired_list += [subj]*h
    # count differences
    cur_count = Counter(arr)
    des_count = Counter(desired_list)
    # remove extras, collect missing
    new_arr = list(arr)
    # remove extras
    for subj in list(cur_count.keys()):
        if cur_count[subj] > des_count.get(subj,0):
            excess = cur_count[subj] - des_count.get(subj,0)
            # remove occurrences from end
            for _ in range(excess):
                idx = new_arr.index(subj)
                new_arr[idx] = None
    # fill missing
    missing = []
    for subj in des_count:
        miss = des_count[subj] - cur_count.get(subj,0)
        missing += [subj]*max(0, miss)
    # fill None positions with missing
    for i in range(len(new_arr)):
        if new_arr[i] is None and missing:
            new_arr[i] = missing.pop()
    # if still left, fill randomly
    for i in range(len(new_arr)):
        if new_arr[i] is None:
            new_arr[i] = random.choice(list(des_count.keys()))
    return new_arr

# --------------------------
# Programação Linear (PL) para balancear carga
# --------------------------
# Dado um candidato, podemos usar PL para ajustar uma distribuição ideal de número de aulas por professor
# A PL aqui visa minimizar a soma das diferenças entre a carga desejada (proporcional) e a carga real.
def linear_balance_teacher_load(chosen_teacher_map, professors_list):
    # chosen_teacher_map: dict (turma,(day,period)) -> teacher_name or None
    # professors_list: list of teacher names
    # Build count per professor
    load = Counter()
    for k,v in chosen_teacher_map.items():
        if v: load[v]+=1
    # Formulate LP: minimize sum |load_p - avg_load| (linearize with aux vars)
    prob = pulp.LpProblem("balance_load", pulp.LpMinimize)
    P = professors_list
    vars_diff = {p: pulp.LpVariable(f"diff_{p}", lowBound=0) for p in P}
    # load_p are constants (observed)
    avg = sum(load.values())/max(1,len(P))
    # minimize sum diffs from average
    prob += pulp.lpSum([vars_diff[p] for p in P])
    # constraints: diff_p >= load_p - avg ; diff_p >= avg - load_p
    for p in P:
        lp = load.get(p,0)
        prob += vars_diff[p] >= lp - avg
        prob += vars_diff[p] >= avg - lp
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    # return diff values as measure (not used to change assignments here; in real system we could reassign to reduce diff)
    diff_vals = {p: vars_diff[p].value() for p in P}
    return diff_vals

# --------------------------
# GA main
# --------------------------
def run_ga(generations=120, pop_size=40):
    pop = initial_population(pop_size)
    profs = example_professors()
    # agora mapeamos subject -> lista de objetos de professor (com classes e availability)
    profs_by_subject = defaultdict(list)
    for p in profs:
        for s in p["subjects"]:
            profs_by_subject[s].append(p)

    scores = [fitness(ind, profs_by_subject) for ind in pop]
    best = pop[scores.index(max(scores))]
    best_score = max(scores)
    for gen in range(generations):
        new_pop = []
        # elitism: carry best 2
        sorted_idx = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
        elite = [copy.deepcopy(pop[sorted_idx[0]]), copy.deepcopy(pop[sorted_idx[1]])]
        new_pop.extend(elite)
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(pop, scores, k=3)
            parent2 = tournament_selection(pop, scores, k=3)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mut_rate=0.03)
            child2 = mutate(child2, mut_rate=0.03)
            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)
        pop = new_pop
        scores = [fitness(ind, profs_by_subject) for ind in pop]
        gen_best_score = max(scores)
        if gen_best_score > best_score:
            best_score = gen_best_score
            best = copy.deepcopy(pop[scores.index(gen_best_score)])
        if gen % 20 == 0:
            print(f"[GA] Geração {gen} melhor score {best_score:.3f}")
    violations, chosen_map = evaluate_resource_violations(best, profs_by_subject)
    prof_names = [p["name"] for p in profs]
    diffs = linear_balance_teacher_load(chosen_map, prof_names)
    return best, chosen_map, violations, diffs

# --------------------------
# Output: gerar Excel
# --------------------------
def schedule_to_dataframe(indiv, chosen_map):
    outputs = {}
    for t in ALL_TURMAS:
        df = pd.DataFrame(index=[f"Aula {p}" for p in range(1, PERIODS_PER_DAY+1)], columns=DAYS)
        for idx, subj in enumerate(indiv[t]):
            day, period = SLOTS[idx]
            prof_name = chosen_map.get((t,(day,period)))
            if prof_name:
                df.at[f"Aula {period}", day] = f"{subj} / {prof_name}"
            else:
                df.at[f"Aula {period}", day] = f"{subj} / SEM PROFESSOR"
        outputs[t] = df
    return outputs

# --------------------------
# Main
# --------------------------
def main():
    print("Executando GA+PL+RSP protótipo...")
    best, chosen_map, violations, diffs = run_ga(generations=120, pop_size=40)
    print("Violations (RSP) detectadas:", violations)
    print("Diferenças de carga (PL): sample", dict(list(diffs.items())[:5]))
    outputs = schedule_to_dataframe(best, chosen_map)
    # salvar excel
    with pd.ExcelWriter("arquivodeaula.xlsx", engine="openpyxl") as writer:
        for turma, df in outputs.items():
            df.to_excel(writer, sheet_name=turma)
        # resumo curriculum
        summary = {t: build_curriculum(t) for t in ALL_TURMAS}
        summary_df = pd.DataFrame(summary).fillna(0).astype(int)
        summary_df.to_excel(writer, sheet_name="Resumo_Carga_Horaria")
        # teacher diff
        diff_df = pd.DataFrame(list(diffs.items()), columns=["Professor","LoadDiff"])
        diff_df.to_excel(writer, sheet_name="Resumo_PL", index=False)
    print("Arquivo salvo: arquivodeaula.xlsx")

if __name__ == "__main__":
    main()