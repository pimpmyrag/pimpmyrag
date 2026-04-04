package rag.connectors.ud.stanza.utils

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.nio.ByteBuffer
import java.nio.LongBuffer

class OnnxTensorHelper {
    companion object {
        private val env = OrtEnvironment.getEnvironment()

        // Pour les tenseurs 1D de type Long
        fun createLongTensor1D(data: LongArray): OnnxTensor {
            val buf = LongBuffer.allocate(data.size)
            buf.put(data)
            buf.flip()
            return OnnxTensor.createTensor(env, buf, longArrayOf(data.size.toLong()))
        }

        // Pour les tenseurs 2D de type Long
        fun createLongTensor2D(data: Array<LongArray>): OnnxTensor {
            val rows = data.size
            val cols = data[0].size
            val buf = LongBuffer.allocate(rows * cols)
            data.forEach { buf.put(it) }
            buf.flip()
            return OnnxTensor.createTensor(env, buf, longArrayOf(rows.toLong(), cols.toLong()))
        }

        // Pour les tenseurs 1D de type Boolean
        fun createBooleanTensor1D(data: BooleanArray): OnnxTensor {
            val buf = ByteBuffer.allocate(data.size)
            data.forEach { buf.put(if (it) 1.toByte() else 0.toByte()) }
            buf.flip()
            return OnnxTensor.createTensor(env, buf, longArrayOf(data.size.toLong()), OnnxJavaType.BOOL)
        }

        // Pour les tenseurs 2D de type Boolean
        fun createBooleanTensor2D(data: Array<BooleanArray>): OnnxTensor {
            val rows = data.size
            val cols = data[0].size
            val buf = ByteBuffer.allocate(rows * cols)
            data.forEach { it.forEach { buf.put(if (it) 1.toByte() else 0.toByte()) } }
            buf.flip()
            return OnnxTensor.createTensor(env, buf, longArrayOf(rows.toLong(), cols.toLong()), OnnxJavaType.BOOL)
        }

        // Pour les tenseurs 3D de type Boolean
        fun createBooleanTensor3D(data: Array<Array<BooleanArray>>): OnnxTensor {
            val dim1 = data.size
            val dim2 = data[0].size
            val dim3 = data[0][0].size
            val buf = ByteBuffer.allocate(dim1 * dim2 * dim3)
            data.forEach { it.forEach { it.forEach { buf.put(if (it) 1.toByte() else 0.toByte()) } } }
            buf.flip()
            return OnnxTensor.createTensor(env, buf, longArrayOf(dim1.toLong(), dim2.toLong(), dim3.toLong()), OnnxJavaType.BOOL)
        }


        // Pour les tenseurs 3D de type Long
        fun createLongTensor3D(data: Array<Array<LongArray>>): OnnxTensor {
            val dim1 = data.size
            val dim2 = data[0].size
            val dim3 = data[0][0].size
            val buf = LongBuffer.allocate(dim1 * dim2 * dim3)
            data.forEach { it.forEach { it.forEach { buf.put(it) } } }
            buf.flip()
            return OnnxTensor.createTensor(env, buf, longArrayOf(dim1.toLong(), dim2.toLong(), dim3.toLong()))
        }

        // Pour les tenseurs 4D de type Long
        fun createLongTensor4D(data: Array<Array<Array<LongArray>>>): OnnxTensor {
            val dim1 = data.size
            val dim2 = data[0].size
            val dim3 = data[0][0].size
            val dim4 = data[0][0][0].size
            val buf = LongBuffer.allocate(dim1 * dim2 * dim3 * dim4)
            data.forEach { it.forEach { it.forEach { it.forEach { buf.put(it) } } } }
            buf.flip()
            return OnnxTensor.createTensor(env, buf, longArrayOf(dim1.toLong(), dim2.toLong(), dim3.toLong(), dim4.toLong()))
        }
    }
}
