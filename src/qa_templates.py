"""Hand-written question templates with paraphrases for all 6 QA categories.

Each category has a list of template strings. During QA generation, one is
randomly selected per example to avoid overfitting to specific phrasings.
"""

# =============================================================================
# Category 1: Intermediate Result Recovery
# =============================================================================

CAT1_SINGLE_THOUGHT = [
    "What numerical result does this computation produce?",
    "What is the intermediate value at this reasoning step?",
    "What number does this calculation yield?",
    "Determine the result of this mathematical step.",
    "What value is computed at this point in the reasoning?",
    "Report the numerical output of this step.",
    "What is the result of this calculation?",
    "What intermediate answer does this step arrive at?",
    "Identify the number produced by this reasoning step.",
    "What quantity does this computation give?",
    "What does this step evaluate to?",
    "State the numerical outcome of this operation.",
]

CAT1_ALL_THOUGHTS = [
    "What are the intermediate results in order?",
    "List all intermediate computation results sequentially.",
    "What numbers are produced at each reasoning step?",
    "Report the intermediate values from first to last.",
    "What sequence of results does this reasoning produce?",
    "Enumerate the intermediate computation outputs.",
    "What values emerge at each step of the reasoning chain?",
    "List the results of each calculation in order.",
    "What are all the intermediate numerical outcomes?",
    "Detail the computation results at each reasoning step.",
]

# =============================================================================
# Category 2: Operation Classification (binary Yes/No)
# =============================================================================

CAT2_MULTIPLICATION = [
    "Does this step involve multiplication?",
    "Is multiplication used in this computation?",
    "Does this calculation include a multiply operation?",
    "Is there a multiplication in this reasoning step?",
    "Does this step perform any multiplication?",
    "Is a product computed in this step?",
    "Does this operation involve multiplying numbers?",
    "Is multiplication part of this calculation?",
    "Does this step include multiplying values together?",
    "Is there a multiplicative operation here?",
]

CAT2_ADDITION = [
    "Does this step involve addition?",
    "Is addition used in this computation?",
    "Does this calculation include an add operation?",
    "Is there an addition in this reasoning step?",
    "Does this step perform any addition?",
    "Is a sum computed in this step?",
    "Does this operation involve adding numbers?",
    "Is addition part of this calculation?",
    "Does this step include adding values together?",
    "Are numbers summed in this step?",
]

CAT2_SUBTRACTION = [
    "Does this step involve subtraction?",
    "Is subtraction used in this computation?",
    "Does this calculation include a subtract operation?",
    "Is there a subtraction in this reasoning step?",
    "Does this step perform any subtraction?",
    "Is a difference computed in this step?",
    "Does this operation involve subtracting numbers?",
    "Is subtraction part of this calculation?",
    "Does this step include subtracting one value from another?",
    "Are numbers subtracted in this step?",
]

CAT2_DIVISION = [
    "Does this step involve division?",
    "Is division used in this computation?",
    "Does this calculation include a divide operation?",
    "Is there a division in this reasoning step?",
    "Does this step perform any division?",
    "Is a quotient computed in this step?",
    "Does this operation involve dividing numbers?",
    "Is division part of this calculation?",
    "Does this step include dividing one value by another?",
    "Are numbers divided in this step?",
]

CAT2_RESULT_GT_100 = [
    "Is the result of this step greater than 100?",
    "Does this computation produce a value above 100?",
    "Is the output of this step more than 100?",
    "Does this calculation yield a result exceeding 100?",
    "Is the intermediate value here larger than 100?",
    "Does this step result in a number greater than one hundred?",
    "Is the computed value above 100?",
    "Does this operation produce an output exceeding 100?",
    "Is the numerical result of this step over 100?",
    "Does this computation give a value greater than 100?",
]

CAT2_USES_PREVIOUS = [
    "Does this step use a result from a previous step?",
    "Is this computation based on a prior intermediate result?",
    "Does this step depend on an earlier calculation?",
    "Is a previously computed value used as input here?",
    "Does this step reference an earlier result?",
    "Is this computation chained from a prior step?",
    "Does this step build on a previous computation?",
    "Is an earlier intermediate result an operand here?",
    "Does this calculation use output from a preceding step?",
    "Is this step dependent on prior reasoning?",
]

CAT2_FIRST_STEP = [
    "Is this the first computation in the reasoning chain?",
    "Is this the initial calculation step?",
    "Does this represent the first arithmetic operation?",
    "Is this where the reasoning computation begins?",
    "Is this the opening step of the calculation sequence?",
    "Is this the starting computation in the chain?",
    "Does this step come first in the reasoning process?",
    "Is this the very first calculation performed?",
    "Is this the beginning of the arithmetic reasoning?",
    "Is this computation the first in the sequence?",
]

CAT2_MULTI_OPERAND = [
    "Does this step involve more than two operands?",
    "Are more than two numbers used in this computation?",
    "Does this calculation use three or more values?",
    "Is this step operating on more than two quantities?",
    "Does this computation involve multiple operands beyond two?",
    "Are there more than two numbers in this operation?",
    "Does this step combine more than two values?",
    "Is this a computation with three or more inputs?",
    "Does this calculation take more than two numbers as input?",
    "Are multiple values beyond two involved in this step?",
]

# Group all Cat 2 sub-types for easy access
CAT2_SUBTYPES = {
    "multiplication": CAT2_MULTIPLICATION,
    "addition": CAT2_ADDITION,
    "subtraction": CAT2_SUBTRACTION,
    "division": CAT2_DIVISION,
    "result_gt_100": CAT2_RESULT_GT_100,
    "uses_previous": CAT2_USES_PREVIOUS,
    "first_step": CAT2_FIRST_STEP,
    "multi_operand": CAT2_MULTI_OPERAND,
}

# =============================================================================
# Category 3: Full Reasoning Description
# =============================================================================

CAT3_FULL_REASONING = [
    "Describe the reasoning steps used to solve this problem.",
    "What sequence of calculations leads to the answer?",
    "Explain the mathematical reasoning encoded here.",
    "Walk through the computation step by step.",
    "Summarize the chain of arithmetic operations performed.",
    "Detail the sequence of calculations in this reasoning.",
    "What steps were taken to arrive at the answer?",
    "Outline the mathematical procedure encoded in these thoughts.",
    "Narrate the reasoning process step by step.",
    "Describe the computation that these thoughts represent.",
    "Explain the calculation chain from start to finish.",
    "What is the sequence of operations that produces the final answer?",
]

# =============================================================================
# Category 4: Problem-Level Property Questions
# =============================================================================

CAT4_NUM_STEPS = [
    "How many arithmetic steps are involved?",
    "How many computation steps does this reasoning use?",
    "What is the total number of calculation steps?",
    "How many separate calculations are performed?",
    "Count the number of arithmetic operations in this reasoning.",
    "How many steps does the computation chain contain?",
    "What is the number of individual calculations?",
    "How many reasoning steps are encoded here?",
    "How many distinct computations are carried out?",
    "What is the step count for this reasoning chain?",
]

CAT4_FINAL_ANSWER = [
    "What is the final answer?",
    "What is the ultimate result of this reasoning?",
    "What answer does this computation arrive at?",
    "What is the final numerical outcome?",
    "What value does the complete reasoning produce?",
    "What is the end result of these calculations?",
    "Report the final answer from this reasoning chain.",
    "What number is the final result?",
    "What is the concluding value of this computation?",
    "State the final answer produced by this reasoning.",
]

CAT4_HAS_SUBTRACTION = [
    "Does this problem require subtraction?",
    "Is subtraction involved in solving this problem?",
    "Does the reasoning chain include any subtraction?",
    "Is a subtract operation used at any point?",
    "Does solving this problem involve subtracting values?",
    "Is there a subtraction step in this reasoning?",
    "Does any step in the chain perform subtraction?",
    "Is subtraction part of the solution process?",
    "Does the computation require subtracting numbers?",
    "Is there a difference calculation in this problem?",
]

CAT4_NEGATIVE_INTERMEDIATE = [
    "Are any intermediate results negative?",
    "Does any step produce a negative value?",
    "Is there a negative number among the intermediate results?",
    "Do any of the computation steps yield a negative outcome?",
    "Are there any negative intermediate values?",
    "Does the reasoning chain include any negative results?",
    "Is a negative number computed at any step?",
    "Are any of the step-by-step results below zero?",
    "Does any calculation produce a negative output?",
    "Are there negative values among the intermediate computations?",
]

CAT4_ANSWER_GT_1000 = [
    "Is the final answer larger than 1000?",
    "Does the final result exceed 1000?",
    "Is the answer greater than one thousand?",
    "Does the computation produce a result above 1000?",
    "Is the ultimate answer more than 1000?",
    "Does the final value surpass 1000?",
    "Is the end result over 1000?",
    "Does the answer come out to more than 1000?",
    "Is the final numerical result above 1000?",
    "Does this problem have an answer exceeding 1000?",
]

CAT4_HAS_PERCENTAGE = [
    "Does this problem involve computing a percentage?",
    "Is a percentage calculation part of this reasoning?",
    "Does the solution involve any percentage operations?",
    "Is percent computation used in solving this?",
    "Does the reasoning chain include percentage calculations?",
    "Is there a percentage-related operation in this problem?",
    "Does solving this involve computing a percent?",
    "Is percentage arithmetic part of the solution?",
    "Does the computation involve any percent-based calculations?",
    "Are percentages used at any point in the reasoning?",
]

CAT4_SUBTYPES = {
    "num_steps": CAT4_NUM_STEPS,
    "final_answer": CAT4_FINAL_ANSWER,
    "has_subtraction": CAT4_HAS_SUBTRACTION,
    "negative_intermediate": CAT4_NEGATIVE_INTERMEDIATE,
    "answer_gt_1000": CAT4_ANSWER_GT_1000,
    "has_percentage": CAT4_HAS_PERCENTAGE,
}

# =============================================================================
# Category 5: Context Prediction
# =============================================================================

CAT5_PREDICT_LATER = [
    "Given these early reasoning steps, what are the results of the remaining steps?",
    "Based on the initial computations, predict the later intermediate results.",
    "From these first steps, what do the subsequent calculations produce?",
    "Using these early thoughts, what are the remaining computation results?",
    "Given the initial reasoning, predict the outcomes of the later steps.",
    "From these beginning steps, what values do the remaining computations yield?",
    "Based on the first steps, forecast the later intermediate values.",
    "Given the early reasoning steps, what are the subsequent results?",
    "From these initial computations, what do the remaining steps produce?",
    "Predict the later computation results based on these early steps.",
]

CAT5_PREDICT_EARLIER = [
    "Given these later reasoning steps, what were the earlier intermediate results?",
    "Based on the final computations, infer the initial intermediate results.",
    "From these last steps, what did the earlier calculations produce?",
    "Using these later thoughts, what were the preceding computation results?",
    "Given the final reasoning, infer the earlier intermediate values.",
    "From these ending steps, what values did the earlier computations yield?",
    "Based on the later steps, reconstruct the earlier intermediate results.",
    "Given the final reasoning steps, what were the initial results?",
    "From these later computations, what did the earlier steps produce?",
    "Infer the earlier computation results based on these later steps.",
]

CAT5_PREDICT_ANSWER = [
    "What is the final answer to this problem?",
    "Based on these reasoning steps, what is the answer?",
    "Given all the computation steps, what is the final result?",
    "From this reasoning, determine the final answer.",
    "Using all the encoded thoughts, what answer do they produce?",
    "What final value does this complete reasoning arrive at?",
    "Based on all the steps, what is the ultimate answer?",
    "Given the full reasoning chain, what is the result?",
    "From all the computation steps, derive the final answer.",
    "What is the answer produced by this complete reasoning?",
]

CAT5_PREDICT_TOPIC = [
    "What type of quantities does this problem involve?",
    "What is the subject matter of this mathematical problem?",
    "What kind of real-world scenario does this problem describe?",
    "What topic or context does this problem relate to?",
    "What type of mathematical scenario is being computed here?",
    "What real-world domain does this computation relate to?",
    "What kind of problem is being solved in this reasoning?",
    "What category of word problem does this represent?",
    "What subject area does this calculation address?",
    "What type of practical problem is this reasoning solving?",
]

# =============================================================================
# Category 6: Thought Informativeness
# =============================================================================

CAT6_MEANINGFUL = [
    "Is this thought encoding a meaningful intermediate result?",
    "Does this thought contain a significant computation result?",
    "Is there a meaningful numerical value encoded in this thought?",
    "Does this thought represent a substantive reasoning step?",
    "Is this a thought that encodes important reasoning information?",
    "Does this thought carry a meaningful intermediate value?",
    "Is there significant computational content in this thought?",
    "Does this thought encode a substantial reasoning result?",
    "Is meaningful reasoning information present in this thought?",
    "Does this thought contain an important computation result?",
]

CAT6_COMPUTATIONAL_VS_TRANSITIONAL = [
    "Is this a computational step or a transitional state?",
    "Does this thought represent a calculation or a transition?",
    "Is this thought performing a computation or serving as a bridge?",
    "Is this a meaningful computation step or a filler thought?",
    "Does this thought encode a calculation or is it transitional?",
    "Is this a step where computation happens or just a transition?",
    "Does this represent active reasoning or a transitional state?",
    "Is this thought a computational node or a connecting state?",
    "Is this where a calculation occurs or is it just passing through?",
    "Does this thought perform arithmetic or serve as a bridge state?",
]
