package io.axes

fun l2(v: FloatArray): Float { var s=0f; for (x in v) s+=x*x; return kotlin.math.sqrt(s) }
fun l2norm(v: FloatArray): FloatArray { val n=l2(v); if (n==0f) return v.copyOf(); return FloatArray(v.size){ i-> v[i]/n } }
fun l2norms(v: List<FloatArray>): List<FloatArray> = v.map { l2norm(it)}

fun add(a: FloatArray,b: FloatArray): FloatArray { val o=FloatArray(a.size); for(i in a.indices) o[i]=a[i]+b[i]; return o }
fun sub(a: FloatArray,b: FloatArray): FloatArray { val o=FloatArray(a.size); for(i in a.indices) o[i]=a[i]-b[i]; return o }
fun mean(list: List<FloatArray>): FloatArray {
    require(list.isNotEmpty())
    val d=list.first().size; val acc=FloatArray(d)
    for (v in list){ require(v.size==d); for(i in 0 until d) acc[i]+=v[i] }
    val inv=1f/list.size; for(i in 0 until d) acc[i]*=inv; return acc
}
fun cosine(a: FloatArray,b: FloatArray): Float {
    var dot=0f; var na=0f; var nb=0f
    for(i in a.indices){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i] }
    if(na==0f||nb==0f) return 0f
    return dot/(kotlin.math.sqrt(na)*kotlin.math.sqrt(nb))
}
