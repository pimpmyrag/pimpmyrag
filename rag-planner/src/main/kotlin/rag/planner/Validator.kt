
package rag.planner

class PlanValidator {
    data class ValidationResult(val ok: Boolean, val reason: String? = null)

    fun validate(plan: ExecutionPlanDag): ValidationResult {
        val stepById = plan.steps.associateBy { it.id }
        for (e in plan.edges) {
            if(e.fromStep == "__seed__") {
                continue
            }
            val src = stepById[e.fromStep] ?: return ValidationResult(false, "Unknown fromStep ${e.fromStep}")
            val dst = stepById[e.toStep]   ?: return ValidationResult(false, "Unknown toStep ${e.toStep}")
            val out = src.outputs.find { it.name == e.fromPort } ?: return ValidationResult(false, "Unknown outPort ${e.fromPort} on ${e.fromStep}")
            val inn = dst.inputs.find { it.name == e.toPort }    ?: return ValidationResult(false, "Unknown inPort ${e.toPort} on ${e.toStep}")
            if (out.type != inn.type) return ValidationResult(false, "Type mismatch ${out.type.simpleName} -> ${inn.type.simpleName} on edge $e")
        }
        val order = topoSort(plan) ?: return ValidationResult(false, "Cycle detected")
        if (order.isEmpty()) return ValidationResult(false, "Empty plan")
        return ValidationResult(true)
    }

    fun topoSort(plan: ExecutionPlanDag): List<DagStep>? {
        val outgoing = plan.steps.associate { it.id to mutableListOf<Edge>() }
        val incoming = plan.steps.associate { it.id to 0 }.toMutableMap()
        plan.edges.forEach { e ->
            if(e.fromStep == "__seed__") {
                return@forEach
            }
            outgoing[e.fromStep]!!.add(e); incoming[e.toStep] = incoming[e.toStep]!! + 1
        }
        val stepById = plan.steps.associateBy { it.id }
        val queue: ArrayDeque<DagStep> = ArrayDeque(plan.steps.filter { incoming[it.id] == 0 })
        val result = mutableListOf<DagStep>()
        while (queue.isNotEmpty()) {
            val s = queue.removeFirst(); result += s
            for (e in outgoing[s.id]!!) {
                incoming[e.toStep] = incoming[e.toStep]!! - 1
                if (incoming[e.toStep] == 0) queue.addLast(stepById[e.toStep]!!)
            }
        }
        return if (result.size == plan.steps.size) result else null
    }
}
