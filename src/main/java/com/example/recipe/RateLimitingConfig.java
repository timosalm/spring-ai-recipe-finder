package com.example.recipe;

import io.github.bucket4j.Bandwidth;
import io.github.bucket4j.Bucket;
import io.github.bucket4j.Refill;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

import jakarta.annotation.PostConstruct;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Configuration for rate limiting to prevent abuse of AI endpoints.
 * Uses Bucket4j for token bucket rate limiting.
 */
@Configuration
public class RateLimitingConfig {

    @Value("${app.rate-limit.recipe-generation.capacity:10}")
    private int recipeGenerationCapacity;

    @Value("${app.rate-limit.recipe-generation.refill-rate:10}")
    private int recipeGenerationRefillRate;

    @Value("${app.rate-limit.recipe-generation.refill-duration-minutes:1}")
    private int recipeGenerationRefillDurationMinutes;

    @Value("${app.rate-limit.file-upload.capacity:5}")
    private int fileUploadCapacity;

    @Value("${app.rate-limit.file-upload.refill-rate:5}")
    private int fileUploadRefillRate;

    @Value("${app.rate-limit.file-upload.refill-duration-minutes:60}")
    private int fileUploadRefillDurationMinutes;

    // In-memory storage for rate limit buckets per IP address
    private final Map<String, Bucket> recipeGenerationBuckets = new ConcurrentHashMap<>();
    private final Map<String, Bucket> fileUploadBuckets = new ConcurrentHashMap<>();

    @PostConstruct
    public void init() {
        // Log configuration on startup
        System.out.println("Rate Limiting Configuration:");
        System.out.println("  Recipe Generation: " + recipeGenerationCapacity + " requests per " + recipeGenerationRefillDurationMinutes + " minute(s)");
        System.out.println("  File Upload: " + fileUploadCapacity + " requests per " + fileUploadRefillDurationMinutes + " minute(s)");
    }

    /**
     * Resolves or creates a bucket for recipe generation rate limiting.
     * 
     * @param key The key (e.g., IP address) to identify the client
     * @return The bucket for this client
     */
    public Bucket resolveRecipeGenerationBucket(String key) {
        return recipeGenerationBuckets.computeIfAbsent(key, k -> createRecipeGenerationBucket());
    }

    /**
     * Resolves or creates a bucket for file upload rate limiting.
     * 
     * @param key The key (e.g., IP address) to identify the client
     * @return The bucket for this client
     */
    public Bucket resolveFileUploadBucket(String key) {
        return fileUploadBuckets.computeIfAbsent(key, k -> createFileUploadBucket());
    }

    /**
     * Creates a new bucket for recipe generation with configured limits.
     */
    private Bucket createRecipeGenerationBucket() {
        Bandwidth limit = Bandwidth.classic(
                recipeGenerationCapacity,
                Refill.intervally(recipeGenerationRefillRate, Duration.ofMinutes(recipeGenerationRefillDurationMinutes))
        );
        return Bucket.builder()
                .addLimit(limit)
                .build();
    }

    /**
     * Creates a new bucket for file upload with configured limits.
     */
    private Bucket createFileUploadBucket() {
        Bandwidth limit = Bandwidth.classic(
                fileUploadCapacity,
                Refill.intervally(fileUploadRefillRate, Duration.ofMinutes(fileUploadRefillDurationMinutes))
        );
        return Bucket.builder()
                .addLimit(limit)
                .build();
    }
}
