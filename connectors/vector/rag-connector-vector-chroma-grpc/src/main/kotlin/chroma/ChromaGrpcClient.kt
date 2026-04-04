
package chroma

import io.grpc.ManagedChannelBuilder

class ChromaGrpcClient(host: String, port: Int) {
    private val channel = ManagedChannelBuilder.forAddress(host, port)
        .usePlaintext()
        .build()

    val vectorStub = VectorReaderGrpc.newBlockingStub(channel)
    val metadataStub = MetadataReaderGrpc.newBlockingStub(channel)
    val querystub = QueryExecutorGrpc.newBlockingStub(channel)
}
