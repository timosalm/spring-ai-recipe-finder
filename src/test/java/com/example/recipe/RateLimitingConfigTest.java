package com.example.recipe;

import io.github.bucket4j.Bucket;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.*;

class RateLimitingConfigTest {

    private RateLimitingConfig rateLimitingConfig;

    @BeforeEach
    void setUp() {
        rateLimitingConfig = new RateLimitingConfig();
        
        // Set test values using reflection
        ReflectionTestUtils.setField(rateLimitingConfig, "recipeGenerationCapacity", 5);
        ReflectionTestUtils.setField(rateLimitingConfig, "recipeGenerationRefillRate", 5);
        ReflectionTestUtils.setField(rateLimitingConfig, "recipeGenerationRefillDurationMinutes", 1);
        ReflectionTestUtils.setField(rateLimitingConfig, "fileUploadCapacity", 2);
        ReflectionTestUtils.setField(rateLimitingConfig, "fileUploadRefillRate", 2);
        ReflectionTestUtils.setField(rateLimitingConfig, "fileUploadRefillDurationMinutes", 60);
    }

    @Test
    void testResolveRecipeGenerationBucket_CreatesBucket() {
        Bucket bucket = rateLimitingConfig.resolveRecipeGenerationBucket("127.0.0.1");
        assertNotNull(bucket);
    }

    @Test
    void testResolveRecipeGenerationBucket_SameKeyReturnsSameBucket() {
        Bucket bucket1 = rateLimitingConfig.resolveRecipeGenerationBucket("127.0.0.1");
        Bucket bucket2 = rateLimitingConfig.resolveRecipeGenerationBucket("127.0.0.1");
        assertSame(bucket1, bucket2);
    }

    @Test
    void testResolveRecipeGenerationBucket_DifferentKeysReturnDifferentBuckets() {
        Bucket bucket1 = rateLimitingConfig.resolveRecipeGenerationBucket("127.0.0.1");
        Bucket bucket2 = rateLimitingConfig.resolveRecipeGenerationBucket("192.168.1.1");
        assertNotSame(bucket1, bucket2);
    }

    @Test
    void testResolveFileUploadBucket_CreatesBucket() {
        Bucket bucket = rateLimitingConfig.resolveFileUploadBucket("127.0.0.1");
        assertNotNull(bucket);
    }

    @Test
    void testResolveFileUploadBucket_SameKeyReturnsSameBucket() {
        Bucket bucket1 = rateLimitingConfig.resolveFileUploadBucket("127.0.0.1");
        Bucket bucket2 = rateLimitingConfig.resolveFileUploadBucket("127.0.0.1");
        assertSame(bucket1, bucket2);
    }

    @Test
    void testRecipeGenerationBucket_RateLimiting() {
        Bucket bucket = rateLimitingConfig.resolveRecipeGenerationBucket("test-ip");
        
        // Should allow 5 requests (capacity)
        for (int i = 0; i < 5; i++) {
            assertTrue(bucket.tryConsume(1), "Request " + (i + 1) + " should be allowed");
        }
        
        // 6th request should be blocked
        assertFalse(bucket.tryConsume(1), "6th request should be blocked");
    }

    @Test
    void testFileUploadBucket_RateLimiting() {
        Bucket bucket = rateLimitingConfig.resolveFileUploadBucket("test-ip");
        
        // Should allow 2 requests (capacity)
        for (int i = 0; i < 2; i++) {
            assertTrue(bucket.tryConsume(1), "Request " + (i + 1) + " should be allowed");
        }
        
        // 3rd request should be blocked
        assertFalse(bucket.tryConsume(1), "3rd request should be blocked");
    }

    @Test
    void testDifferentClientsHaveSeparateLimits() {
        Bucket bucket1 = rateLimitingConfig.resolveRecipeGenerationBucket("client1");
        Bucket bucket2 = rateLimitingConfig.resolveRecipeGenerationBucket("client2");
        
        // Exhaust client1's limit
        for (int i = 0; i < 5; i++) {
            assertTrue(bucket1.tryConsume(1));
        }
        assertFalse(bucket1.tryConsume(1), "Client1 should be rate limited");
        
        // Client2 should still have capacity
        assertTrue(bucket2.tryConsume(1), "Client2 should not be rate limited");
    }
}
