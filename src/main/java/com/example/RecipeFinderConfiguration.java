package com.example;

import com.sap.ai.sdk.foundationmodels.openai.OpenAiClient;
import com.sap.ai.sdk.foundationmodels.openai.OpenAiModel;
import com.sap.ai.sdk.foundationmodels.openai.OpenAiUserMessage;
import com.sap.ai.sdk.foundationmodels.openai.spring.OpenAiSpringEmbeddingModel;
import com.sap.ai.sdk.orchestration.OrchestrationModuleConfig;
import com.sap.ai.sdk.orchestration.spring.OrchestrationChatModel;
import com.sap.ai.sdk.orchestration.spring.OrchestrationChatOptions;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.model.openai.autoconfigure.OpenAiChatProperties;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.Resource;

import java.util.Optional;

import static com.sap.ai.sdk.orchestration.OrchestrationAiModel.GPT_4O;
import static com.sap.ai.sdk.orchestration.OrchestrationAiModel.Parameter.TEMPERATURE;

@Configuration
class RecipeFinderConfiguration {

    @Value("classpath:/prompts/fix-json-response")
    private Resource fixJsonResponsePromptResource;

    @Bean
    ChatClient chatClient(ChatClient.Builder chatClientBuilder, Optional<ChatOptions> chatOptions) {
		chatOptions.ifPresent(chatClientBuilder::defaultOptions);
        return chatClientBuilder.defaultSystem(fixJsonResponsePromptResource).build();
    }

    @Profile("sap")
    @Bean
    ChatModel chatModel() {
        return new OrchestrationChatModel();
    }

    @Profile("sap")
    @Bean
    ChatOptions chatOptions(@Value("${spring.ai.openai.chat.options.temperature}") Double temperature) {
        var config = new OrchestrationModuleConfig().withLlmConfig(GPT_4O.withParam(TEMPERATURE, temperature));
        return new OrchestrationChatOptions(config);
    }

    @Profile("sap")
    @Bean
    EmbeddingModel embeddingModel() {
        var client = OpenAiClient.forModel(OpenAiModel.TEXT_EMBEDDING_3_SMALL);
        return new OpenAiSpringEmbeddingModel(client);
    }

    @ConditionalOnMissingBean(VectorStore.class)
    @Bean
    VectorStore simpleVectorStore(EmbeddingModel embeddingModel) {
        return SimpleVectorStore.builder(embeddingModel).build();
    }
}